import torch
import torch.nn.functional as F
from copy import deepcopy

from .incremental_learning import Incremental_Learning_Approach


class Appr(Incremental_Learning_Approach):
    """LwF Hydra v9 — identical to lwf_hydra_v5 (CE on cls only, KD every head from itself), lamb = 1/6.

    6 KD terms x 1/6 vs 1 CE term  ->  CE:KD = 1:1 (as plain lwf).

    Differences from lwf_hydra_v3:
      - current task: CE with ground truth on the cls head only (as lwf_hydra_v2)
      - old tasks: every head distils from its OWN head of the old model (as v3)
      - eval: cls head only

    Approach-specific args (approach_args in config):
        lamb  (float, default 1/6) — distillation loss weight
        T     (int,   default 2)   — softmax temperature
    """

    def __init__(self, args, model, logger=None, exemplars_dataset=None):
        super().__init__(args, model, logger, exemplars_dataset)
        self.model_old = None
        aargs = args.get('approach_args', {})
        self.lamb = aargs.get('lamb', 1/6)
        self.T    = aargs.get('T', 2)

    def post_train_process(self, t, trn_loader):
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
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

    def eval(self, t, val_loader):
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                targets_old = None
                if t > 0:
                    targets_old = self.model_old(images.to(self.device))
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
                hits_taw, hits_tag = self.calculate_metrics(outputs["cls_logits"], targets)
                total_loss    += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num     += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def load_progress(self, results_path, task):
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
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
        loss = 0
        all_heads = [outputs["cls_logits"]] + outputs["dist_logits"]

        if t > 0:
            # every head distils from its own head of the old model
            all_heads_old = [targets_old["cls_logits"]] + targets_old["dist_logits"]
            for head, head_old in zip(all_heads, all_heads_old):
                loss += self.lamb * self.cross_entropy(
                    torch.cat(head[:t], dim=1), torch.cat(head_old[:t], dim=1), exp=1.0 / self.T)

        # CE on current task — cls head only
        loss += F.cross_entropy(outputs["cls_logits"][t], targets - self.model.task_offset[t])

        return loss
