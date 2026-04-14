import torch
import torch.nn.functional as F
from copy import deepcopy

from .incremental_learning import Incremental_Learning_Approach
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Incremental_Learning_Approach):
    """
    LwF with soft KL distillation on both cls and dist logits — cls-only inference.
    Ported to the v2 args-dict interface. Requires LLL_Net_Distilled (distilled: true).

    - cls head, old tasks : soft KL distillation against teacher cls_logits
    - dist head, old tasks: soft KL distillation against teacher cls_logits (same teacher target)
    - current task: CE with ground truth on cls head only
    - inference: cls logits only

    Both heads are anchored to the same teacher signal (cls logits), giving the dist token
    an additional regularisation path that reinforces backbone stability.
    """

    def __init__(self, args, model, logger=None, exemplars_dataset=None):
        super().__init__(args, model, logger, exemplars_dataset)
        self.model_old = None
        aargs = args.get('approach_args', {})
        self.lamb = aargs.get('lamb', 1.0)
        self.T    = aargs.get('T', 2)

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

    def train_epoch(self, t, trn_loader):
        self.model.train()
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
        """Soft cross-entropy with optional temperature scaling."""
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

    def criterion(self, t, outputs, targets, outputs_old):
        loss = 0
        cls_logits  = outputs["cls_logits"]
        dist_logits = outputs["dist_logits"]

        if t > 0:
            teacher_cls = torch.cat(outputs_old["cls_logits"][:t], dim=1)

            # Loss 1: soft KL on cls logits for old tasks
            loss += self.lamb * self.cross_entropy(
                torch.cat(cls_logits[:t], dim=1),
                teacher_cls,
                exp=1.0 / self.T
            )

            # Loss 2: soft KL on dist logits for old tasks (same teacher target)
            loss += self.lamb * self.cross_entropy(
                torch.cat(dist_logits[:t], dim=1),
                teacher_cls,
                exp=1.0 / self.T
            )

        # Loss 3: CE with GT on current task — cls head only
        loss += F.cross_entropy(cls_logits[t], targets - self.model.task_offset[t])

        return loss
