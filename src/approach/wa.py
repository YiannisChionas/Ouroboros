import torch
import torch.nn.functional as F
from copy import deepcopy

from .incremental_learning import Incremental_Learning_Approach
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Incremental_Learning_Approach):
    """Weight Aligning (WA)
    https://arxiv.org/abs/1911.07053

    LwF with an auto-scheduled KD weight (lambda = known/total) and weight alignment
    after each task to correct the recency bias in new head weights.

    Weight alignment: rescales the new head's weights so that their mean L2 norm
    matches the mean L2 norm of the old heads, removing the magnitude imbalance
    that causes the model to over-predict new classes.

    Approach-specific args are read from args['approach_args']:
        T    (int,   default 2)   — softmax temperature for KD
        lamb (float, default -1)  — KD weight; -1 = known/total auto schedule
    """

    def __init__(self, args, model, logger=None, exemplars_dataset=None):
        super().__init__(args, model, logger, exemplars_dataset)
        self.model_old = None
        aargs = args.get('approach_args', {})
        self.T    = aargs.get('T', 2)
        self.lamb = aargs.get('lamb', -1)

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    def _get_optimizer(self):
        if self.exemplars_dataset is None and len(self.model.heads) > 1:
            params_all = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params_all = list(self.model.parameters())
        params = [p for p in params_all if p.requires_grad]
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_loop(self, t, trn_loader, val_loader):
        if self.exemplars_dataset is not None and t > 0:
            trn_loader = torch.utils.data.DataLoader(
                trn_loader.dataset + self.exemplars_dataset,
                batch_size=trn_loader.batch_size, shuffle=True,
                num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)

        super().train_loop(t, trn_loader, val_loader)

        if self.exemplars_dataset is not None:
            self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        if t > 0:
            self._apply_weight_aligning(t)

        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def _apply_weight_aligning(self, t):
        """Rescale new head weights so their mean L2 norm matches the old heads."""
        with torch.no_grad():
            norms_old = torch.cat([torch.norm(self.model.heads[i].weight, p=2, dim=1) for i in range(t)])
            mean_old  = norms_old.mean()
            mean_new  = torch.norm(self.model.heads[t].weight, p=2, dim=1).mean()
            gamma = mean_old / mean_new
            self.model.heads[t].weight.data *= gamma
            print(f"WA: gamma={gamma:.4f}  (mean_norm_old={mean_old:.4f}, mean_norm_new={mean_new:.4f})")

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
            self.optimizer.step()

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
        # Lambda schedule: auto (known/total) or fixed
        if t > 0:
            lamb = (self.model.task_cls[:t].sum().float() / self.model.task_cls.sum()).to(self.device) \
                   if self.lamb == -1 else self.lamb

        # CE loss — with exemplars use all heads (global labels), else task head only
        if self.exemplars_dataset is not None:
            loss_ce = F.cross_entropy(torch.cat(outputs, dim=1), targets)
        else:
            loss_ce = F.cross_entropy(outputs[t], targets - self.model.task_offset[t])

        if t == 0:
            return loss_ce

        # KD loss over old tasks
        loss_kd = self._kd_loss(
            torch.cat(outputs[:t], dim=1),
            torch.cat(outputs_old[:t], dim=1),
            self.T,
        )

        return (1.0 - lamb) * loss_ce + lamb * loss_kd

    @staticmethod
    def _kd_loss(pred, soft, T):
        """KL-divergence KD loss with temperature scaling."""
        pred = F.log_softmax(pred / T, dim=1)
        soft = F.softmax(soft / T, dim=1)
        return -(soft * pred).sum(dim=1).mean()
