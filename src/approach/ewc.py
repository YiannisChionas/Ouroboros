import os
import warnings
import torch
import itertools

from .incremental_learning import Incremental_Learning_Approach
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Incremental_Learning_Approach):
    """Elastic Weight Consolidation (EWC)
    http://arxiv.org/abs/1612.00796
    """

    def __init__(self, args, model, logger=None, exemplars_dataset=None):
        super().__init__(args, model, logger, exemplars_dataset)
        aargs = args.get('approach_args', {})
        self.lamb = aargs.get('lamb', 5000)
        self.alpha = aargs.get('alpha', 0.5)
        self.sampling_type = aargs.get('fi_sampling_type', 'max_pred')
        self.num_samples = aargs.get('fi_num_samples', -1)

        # Importance weights are kept only for the backbone, not the heads
        feat_ext = self.model.model
        self.older_params = {n: p.clone().detach().to(self.device)
                             for n, p in feat_ext.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros(p.shape).to(self.device)
                       for n, p in feat_ext.named_parameters() if p.requires_grad}

    def _get_optimizer(self):
        """Returns the optimizer"""
        if not self.exemplars_dataset and len(self.model.heads) > 1:
            # if there are no exemplars, previous heads are not modified
            params_all = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params_all = list(self.model.parameters())
        params = [p for p in params_all if p.requires_grad]
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if self.exemplars_dataset and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        super().train_loop(t, trn_loader, val_loader)

        if self.exemplars_dataset:
            self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Store current parameters and update Fisher information after each task."""

        # Store current parameters for the next task
        backbone = getattr(self.model.model, 'module', self.model.model)
        self.older_params = {n: p.clone().detach().to(self.device)
                             for n, p in backbone.named_parameters() if p.requires_grad}

        # Compute and merge Fisher information
        curr_fisher = self.compute_fisher_matrix_diag(trn_loader)
        for n in self.fisher.keys():
            if self.alpha == -1:
                # Accumulate fisher over time weighted by proportion of classes seen so far
                alpha = (sum(self.model.task_cls[:t]) / sum(self.model.task_cls)).to(self.device)
                self.fisher[n] = alpha * self.fisher[n] + (1 - alpha) * curr_fisher[n]
            else:
                self.fisher[n] = self.alpha * self.fisher[n] + (1 - self.alpha) * curr_fisher[n]

    def compute_fisher_matrix_diag(self, trn_loader):
        """Compute diagonal Fisher information matrix via gradient accumulation."""
        backbone = getattr(self.model.model, 'module', self.model.model)
        fisher = {n: torch.zeros(p.shape).to(self.device)
                  for n, p in backbone.named_parameters() if p.requires_grad}
        n_samples_batches = (self.num_samples // trn_loader.batch_size + 1) if self.num_samples > 0 \
            else (len(trn_loader.dataset) // trn_loader.batch_size)
        self.model.train()
        for images, targets in itertools.islice(trn_loader, n_samples_batches):
            outputs = self.model.forward(images.to(self.device))

            if self.sampling_type == 'true':
                preds = targets.to(self.device)
            elif self.sampling_type == 'max_pred':
                preds = torch.cat(outputs, dim=1).argmax(1).flatten()
            elif self.sampling_type == 'multinomial':
                probs = torch.nn.functional.softmax(torch.cat(outputs, dim=1), dim=1)
                preds = torch.multinomial(probs, len(targets)).flatten()

            loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), preds)
            self.optimizer.zero_grad()
            loss.backward()
            for n, p in backbone.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2) * len(targets)

        n_samples = n_samples_batches * trn_loader.batch_size
        fisher = {n: (p / n_samples) for n, p in fisher.items()}
        return fisher

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        loss = 0
        if t > 0:
            loss_reg = 0
            backbone = getattr(self.model.model, 'module', self.model.model)
            for n, p in backbone.named_parameters():
                if n in self.fisher.keys():
                    loss_reg += torch.sum(self.fisher[n] * (p - self.older_params[n]).pow(2)) / 2
            loss += self.lamb * loss_reg
        if self.exemplars_dataset:
            return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])

    def save_progress(self, results_path, task):
        """Save Fisher matrix and older_params for resume."""
        torch.save(
            {'fisher':       {k: v.cpu() for k, v in self.fisher.items()},
             'older_params': {k: v.cpu() for k, v in self.older_params.items()}},
            os.path.join(results_path, f"task{task}_ewc.pth")
        )

    def load_progress(self, results_path, task):
        """Restore Fisher matrix and older_params on resume."""
        ewc_file = os.path.join(results_path, f"task{task}_ewc.pth")
        if not os.path.isfile(ewc_file):
            warnings.warn(f"EWC file NOT found at {ewc_file}!")
            return
        st = torch.load(ewc_file, map_location=self.device, weights_only=False)
        self.fisher       = {k: v.to(self.device) for k, v in st['fisher'].items()}
        self.older_params = {k: v.to(self.device) for k, v in st['older_params'].items()}
        print(f"Fisher: {len(self.fisher)} layers | older_params: {len(self.older_params)} layers loaded")
