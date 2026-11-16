import timm.optim
import torch
import torch.nn.functional as F
from copy import deepcopy

from .incremental_learning import Incremental_Learning_Approach

class Appr(Incremental_Learning_Approach):
    """LwF v3 — lwf_v2 with the bottom of the ViT frozen for the whole run (from task 0).
    Frozen: patch_embed, cls_token, pos_embed and the first `freeze_blocks` transformer blocks.
    The upper blocks, the final norm and the heads train as in lwf_v2 (CE on the current task + lamb * KD).

    Approach-specific args are read from args['approach_args']:
        lamb          (float, default 1.0) — distillation loss weight
        T             (int,   default 2)   — softmax temperature
        freeze_blocks (int,   default 6)   — number of bottom transformer blocks to freeze
    """

    def __init__(self, args, model, logger=None, exemplars_dataset=None):
        super().__init__(args, model, logger, exemplars_dataset)
        self.model_old = None
        aargs = args.get('approach_args', {})
        self.lamb = aargs.get('lamb', 1.0)
        self.T    = aargs.get('T', 2)
        self.freeze_blocks = aargs.get('freeze_blocks', 6)

    def _freeze_bottom(self):
        """Freeze the embeddings and the first freeze_blocks blocks of the ViT backbone."""
        vit = self.model.model
        frozen = [vit.patch_embed] + list(vit.blocks[:self.freeze_blocks])
        for module in frozen:
            for p in module.parameters():
                p.requires_grad = False
        vit.cls_token.requires_grad = False
        vit.pos_embed.requires_grad = False

    def _get_optimizer(self):
        """Same as the base optimizer, but passes nesterov only to SGD (Adam rejects it).
        Called by train_loop before every task, so the bottom is frozen before the optimizer is built."""
        self._freeze_bottom()
        params = [p for p in self.model.parameters() if p.requires_grad]
        extra = {'nesterov': False} if self.optimizer_name == 'sgd' else {}
        return timm.optim.create_optimizer_v2(
            params,
            opt=self.optimizer_name,
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            **extra,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def post_train_process(self, t, trn_loader):
        """Save a frozen copy of the model after each task — used as teacher for next task."""
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        for images, targets in trn_loader:
            targets_old = None
            if t > 0:
                with torch.no_grad():
                    targets_old = self.model_old(images.to(self.device))
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping)
            self.optimizer.step()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                targets_old = None
                if t > 0:
                    targets_old = self.model_old(images.to(self.device))
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device), targets_old)

                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)

                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)

        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def load_progress(self, results_path, task):
        """Restore model_old from the current (already-loaded) model state."""
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Cross-entropy with temperature scaling — used for knowledge distillation."""
        out = F.softmax(outputs, dim=1)
        tar = F.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def criterion(self, t, outputs, targets, outputs_old=None):
        """Returns the loss value"""
        loss = 0
        if t > 0:
            # Knowledge distillation loss over all previous tasks
            loss += self.lamb * self.cross_entropy(torch.cat(outputs[:t], dim=1),
                                                   torch.cat(outputs_old[:t], dim=1),
                                                   exp=1.0 / self.T)
        # Current task cross-entropy
        loss += F.cross_entropy(outputs[t], targets - self.model.task_offset[t])
        return loss