import warnings
import torch
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader

from .incremental_learning import Incremental_Learning_Approach
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Incremental_Learning_Approach):
    """End-to-End Incremental Learning (EEIL)
    http://openaccess.thecvf.com/content_ECCV_2018/papers/Francisco_M._Castro_End-to-End_Incremental_Learning_ECCV_2018_paper.pdf

    LwF with exemplars and two-phase training per task:
      Phase 1 — Unbalanced: train on current task + exemplars with full lr/epochs + KD on old heads
      Phase 2 — Balanced finetuning: subsample current task to match exemplar count, reduced lr/epochs + KD on all old heads

    Approach-specific args are read from args['approach_args']:
        lamb                 (float, default 1.0)  — KD loss weight
        T                    (int,   default 2)    — softmax temperature
        lr_finetuning_factor (float, default 0.01) — lr multiplier for balanced phase
        nepochs_finetuning   (int,   default 40)   — epochs for balanced phase
        noise_grad           (bool,  default False) — add gaussian noise to gradients
    """

    def __init__(self, args, model, logger=None, exemplars_dataset=None):
        super().__init__(args, model, logger, exemplars_dataset)
        self.model_old = None
        aargs = args.get('approach_args', {})
        self.lamb                 = aargs.get('lamb', 1.0)
        self.T                    = aargs.get('T', 2)
        self.lr_finetuning_factor = aargs.get('lr_finetuning_factor', 0.01)
        self.nepochs_finetuning   = aargs.get('nepochs_finetuning', 40)
        self.noise_grad           = aargs.get('noise_grad', False)

        self._train_epoch         = 0
        self._finetuning_balanced = False

        if not self.exemplars_dataset:
            warnings.warn("EEIL requires exemplars. Check exemplars_args in config.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def post_train_process(self, t, trn_loader):
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_loop(self, t, trn_loader, val_loader):
        if t == 0:
            super().train_loop(t, trn_loader, val_loader)
            loader = trn_loader
        else:
            loader = self._train_unbalanced(t, trn_loader, val_loader)
            self._train_balanced(t, trn_loader, val_loader)

        self.exemplars_dataset.collect_exemplars(self.model, loader, val_loader.dataset.transform)

    def _train_unbalanced(self, t, trn_loader, val_loader):
        self._finetuning_balanced = False
        self._train_epoch = 0
        loader = self._get_train_loader(trn_loader, balanced=False)
        super().train_loop(t, loader, val_loader)
        return loader

    def _train_balanced(self, t, trn_loader, val_loader):
        self._finetuning_balanced = True
        self._train_epoch = 0
        orig_lr      = self.lr
        orig_nepochs = self.nepochs
        self.lr      = orig_lr * self.lr_finetuning_factor
        self.nepochs = self.nepochs_finetuning
        loader = self._get_train_loader(trn_loader, balanced=True)
        super().train_loop(t, loader, val_loader)
        self.lr      = orig_lr
        self.nepochs = orig_nepochs

    def _get_train_loader(self, trn_loader, balanced=False):
        """Combine current task data with exemplars. If balanced, subsample current task to exemplar count."""
        current_ds = trn_loader.dataset
        if balanced:
            n = len(self.exemplars_dataset)
            indices = torch.randperm(len(current_ds))[:n]
            current_ds = torch.utils.data.Subset(current_ds, indices)
        ds = self.exemplars_dataset + current_ds
        return DataLoader(ds, batch_size=trn_loader.batch_size, shuffle=True,
                          num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)

    def train_epoch(self, t, trn_loader):
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            outputs_old = None
            if t > 0:
                with torch.no_grad():
                    outputs_old = self.model_old(images.to(self.device))
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping)
            if self.noise_grad:
                self._add_noise_grad(self.model.parameters(), self._train_epoch)
            self.optimizer.step()
        self._train_epoch += 1

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval(self, t, val_loader):
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                outputs_old = None
                if t > 0:
                    outputs_old = self.model_old(images.to(self.device))
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                total_loss    += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num     += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def save_progress(self, results_path, task):
        import os
        if self.exemplars_dataset is not None:
            torch.save({'images': self.exemplars_dataset.images, 'labels': self.exemplars_dataset.labels},
                       os.path.join(results_path, f"task{task}_exemplars.pth"))

    def load_progress(self, results_path, task):
        import os
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

        ex_file = os.path.join(results_path, f"task{task}_exemplars.pth")
        if os.path.isfile(ex_file) and self.exemplars_dataset is not None:
            state = torch.load(ex_file, weights_only=False)
            self.exemplars_dataset.images = state['images']
            self.exemplars_dataset.labels = state['labels']
            print(f"Loaded {len(self.exemplars_dataset.images)} exemplars from {ex_file}")

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def criterion(self, t, outputs, targets, outputs_old=None):
        # CE on all heads (exemplars have global labels)
        loss = F.cross_entropy(torch.cat(outputs, dim=1), targets)

        if t > 0 and outputs_old is not None:
            # During balanced finetuning apply KD to all old heads (0..t-1),
            # during unbalanced only to heads 0..t-2.
            last_head_idx = t if self._finetuning_balanced else (t - 1)
            for i in range(last_head_idx):
                loss += self.lamb * F.binary_cross_entropy(
                    F.softmax(outputs[i] / self.T, dim=1),
                    F.softmax(outputs_old[i] / self.T, dim=1),
                )

        return loss

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_noise_grad(self, parameters, iteration, eta=0.3, gamma=0.55):
        """Add Gaussian noise to gradients for regularization (Section 4.2 of paper)."""
        parameters = [p for p in parameters if p.grad is not None]
        variance = eta / ((1 + iteration) ** gamma)
        for p in parameters:
            p.grad.add_(torch.randn(p.grad.shape, device=p.grad.device) * variance)
