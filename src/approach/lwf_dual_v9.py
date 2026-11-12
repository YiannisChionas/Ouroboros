import time
import timm.optim
import torch
import torch.nn.functional as F
from copy import deepcopy

from .incremental_learning import Incremental_Learning_Approach

class Appr(Incremental_Learning_Approach):
    """LwF Dual v9 — lwf_dual_v3 (plain LwF on the cls head ONLY of DeiT) with the lwf_v2 recipe.

      - current task: CE with ground truth on the cls head only
      - old tasks: KD of the cls head only, from the OLD CLS head   -> CE:KD = 1:6 (lamb 6)
      - the dist head exists in the network but is NOT used in the loss
      - recipe (config): Adam lr 1e-4 -> 1e-6 cosine, no weight decay
      - head warm-up: 5 epochs, CE on the new cls head only, lr x10, backbone not updated
      - eval: cls head only

    Approach-specific args are read from args['approach_args']:
        lamb  (float, default 6.0) — distillation loss weight
        T     (int,   default 2)   — softmax temperature
    """

    def __init__(self, args, model, logger=None, exemplars_dataset=None):
        super().__init__(args, model, logger, exemplars_dataset)
        self.model_old = None
        aargs = args.get('approach_args', {})
        self.lamb = aargs.get('lamb', 6.0)
        self.T    = aargs.get('T', 2)

    def _get_optimizer(self):
        """Same as the base optimizer, but passes nesterov only to SGD (Adam rejects it)."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        extra = {'nesterov': False} if self.optimizer_name == 'sgd' else {}
        return timm.optim.create_optimizer_v2(params, opt=self.optimizer_name, lr=self.lr,
                                              weight_decay=self.weight_decay, momentum=self.momentum, **extra)

    def pre_train_process(self, task, train_loader):
        """Head warm-up (task > 0): only the new head(s) self.model.heads[-1] are trained with CE, backbone untouched.
        The base warm-up cannot be used here: it reads outputs[task] (DeiT returns a dict) and trains heads[-1] only."""
        if not (self.warmup_epochs and task > 0):
            return
        new_heads = [self.model.heads[-1]]
        params = [p for h in new_heads for p in h.parameters()]
        self.optimizer = timm.optim.create_optimizer_v2(params, opt=self.optimizer_name, lr=self.warmup_lr_factor,
                                                        weight_decay=self.weight_decay, momentum=self.momentum)
        offset = self.model.task_offset[task]
        for epoch in range(self.warmup_epochs):
            clock0 = time.time()
            for h in new_heads:
                h.train()
            total_loss, total_hits, total_num = 0.0, 0, 0
            for images, targets in train_loader:
                targets = targets.to(self.device) - offset
                outputs = self.model(images.to(self.device))
                loss = sum(F.cross_entropy(outputs[k][task], targets) for k in ("cls_logits",))
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.clipping)
                self.optimizer.step()
                total_loss += loss.item() * len(targets)
                total_hits += (outputs["cls_logits"][task].argmax(1) == targets).sum().item()
                total_num += len(targets)
            print('| Warm-up Epoch {:3d}, time={:5.1f}s | Train: loss={:.3f}, cls TAw acc={:5.1f}% |'.format(
                epoch + 1, time.time() - clock0, total_loss / total_num, 100 * total_hits / total_num))
            self.logger.log_scalar(task=task, iter=epoch + 1, name="loss", value=total_loss / total_num, group="warmup")
            self.logger.log_scalar(task=task, iter=epoch + 1, name="acc", value=100 * total_hits / total_num, group="warmup")

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

                # Inference: cls logits only
                hits_taw, hits_tag = self.calculate_metrics(outputs["cls_logits"], targets)

                total_loss    += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num     += len(targets)

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

    def criterion(self, t, outputs, targets, targets_old):
        """Returns the loss value"""
        loss = 0
        cls_logits = outputs["cls_logits"]

        if t > 0:
            # KD on cls logits for old tasks, from the old cls head
            loss += self.lamb * self.cross_entropy(
                torch.cat(cls_logits[:t], dim=1),
                torch.cat(targets_old["cls_logits"][:t], dim=1),
                exp=1.0 / self.T
            )

        # CE with GT on current task — cls head only
        loss += F.cross_entropy(cls_logits[t], targets - self.model.task_offset[t])

        return loss
