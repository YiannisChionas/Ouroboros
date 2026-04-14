import torch
import torch.nn.functional as F
from copy import deepcopy
from argparse import ArgumentParser

from .inc_learn import Incremental_Learning_Approach
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Incremental_Learning_Approach):
    """
    LwF with cosine feature distillation on the dist token.

    - cls head, old tasks : soft KL distillation on logits (teacher cls_logits)
    - dist token, old tasks: cosine similarity loss on backbone features (student dist_feat vs teacher dist_feat)
    - current task (cls + dist): CE with ground truth
    - inference: avg(cls_logits, dist_logits)

    Motivation: constraining the dist token at feature level preserves the geometry
    of the representation space, complementing the logit-level KL on the cls head.
    """

    def __init__(self,
                 model,
                 device,
                 nepochs=100,
                 lr_scheduler='none',
                 lr=0.05,
                 lr_min=None,
                 lr_factor=None,
                 lr_patience=None,
                 clipgrad=10000,
                 momentum=0,
                 wd=0,
                 multi_softmax=False,
                 wu_nepochs=0,
                 wu_lr_factor=1,
                 fix_bn=False,
                 freeze_backbone=False,
                 eval_on_train=False,
                 logger=None,
                 exemplars_dataset=None,
                 lamb=1,
                 lamb_cos=1,
                 T=2):
        super(Appr, self).__init__(model,
                                   device,
                                   nepochs,
                                   lr_scheduler,
                                   lr,
                                   lr_min,
                                   lr_factor,
                                   lr_patience,
                                   clipgrad,
                                   momentum,
                                   wd,
                                   multi_softmax,
                                   wu_nepochs,
                                   wu_lr_factor,
                                   fix_bn,
                                   freeze_backbone,
                                   eval_on_train,
                                   logger,
                                   exemplars_dataset)
        self.model_old = None
        self.lamb = lamb
        self.lamb_cos = lamb_cos
        self.T = T

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Weight for KL distillation loss on cls logits (default=%(default)s)')
        parser.add_argument('--lamb-cos', default=1, type=float, required=False,
                            help='Weight for cosine feature loss on dist token (default=%(default)s)')
        parser.add_argument('--T', default=2, type=int, required=False,
                            help='Temperature for KL distillation (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

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
                    outputs_old = self.model_old(images.to(self.device), return_features=True)
            outputs = self.model(images.to(self.device), return_features=True)
            loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def eval(self, t, val_loader):
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            stats = {i: {'cls_max': 0, 'dist_max': 0, 'cls_norm': 0, 'dist_norm': 0, 'count': 0} for i in range(t + 1)}
            feat_cos_sim_sum, feat_norm_student_sum, feat_norm_teacher_sum, feat_count = 0, 0, 0, 0

            self.model.eval()
            for images, targets in val_loader:
                outputs_old = None
                if t > 0:
                    outputs_old = self.model_old(images.to(self.device), return_features=True)
                outputs = self.model(images.to(self.device), return_features=True)
                loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)

                for task_id in range(t + 1):
                    c_out = outputs["cls_logits"][task_id]
                    d_out = outputs["dist_logits"][task_id]
                    stats[task_id]['cls_max'] += c_out.max(dim=1)[0].sum().item()
                    stats[task_id]['dist_max'] += d_out.max(dim=1)[0].sum().item()
                    stats[task_id]['cls_norm'] += torch.norm(c_out, p=2, dim=1).sum().item()
                    stats[task_id]['dist_norm'] += torch.norm(d_out, p=2, dim=1).sum().item()
                    stats[task_id]['count'] += len(targets)

                if t > 0:
                    s_feat = outputs["dist_features"]
                    t_feat = outputs_old["dist_features"]
                    feat_cos_sim_sum += F.cosine_similarity(s_feat, t_feat, dim=-1).sum().item()
                    feat_norm_student_sum += torch.norm(s_feat, p=2, dim=-1).sum().item()
                    feat_norm_teacher_sum += torch.norm(t_feat, p=2, dim=-1).sum().item()
                    feat_count += s_feat.shape[0]

                eval_outputs = [
                    (cls_out + dist_out) / 2
                    for cls_out, dist_out in zip(outputs["cls_logits"], outputs["dist_logits"])
                ]
                hits_taw, hits_tag = self.calculate_metrics(eval_outputs, targets)

                total_loss += loss.data.cpu().numpy().item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)

            print("\n" + "=" * 40)
            print(f"DEBUG: Logit Statistics at Task {t}")
            for task_id in range(t + 1):
                s = stats[task_id]
                c = s['count']
                print(f"Task {task_id}:")
                print(f"  CLS  -> AvgMax: {s['cls_max']/c:.2f}, AvgNorm: {s['cls_norm']/c:.2f}")
                print(f"  DIST -> AvgMax: {s['dist_max']/c:.2f}, AvgNorm: {s['dist_norm']/c:.2f}")
            if t > 0:
                print(f"Dist Feature Stats (student vs teacher):")
                print(f"  AvgCosSim : {feat_cos_sim_sum / feat_count:.4f}  (1.0 = identical, 0.0 = orthogonal)")
                print(f"  AvgNorm student: {feat_norm_student_sum / feat_count:.4f}")
                print(f"  AvgNorm teacher: {feat_norm_teacher_sum / feat_count:.4f}")
            print("=" * 40 + "\n")

        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Soft cross-entropy with optional temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
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

        cls_logits = outputs["cls_logits"]
        dist_logits = outputs["dist_logits"]

        if t > 0:
            # --- Loss 1: soft KL on cls logits for old tasks ---
            loss += self.lamb * self.cross_entropy(
                torch.cat(cls_logits[:t], dim=1),
                torch.cat(outputs_old["cls_logits"][:t], dim=1),
                exp=1.0 / self.T
            )

            # --- Loss 2: cosine similarity on dist token features ---
            # Constrains the backbone representation of the dist token
            # independently of the task heads.
            student_dist = F.normalize(outputs["dist_features"], dim=-1)
            teacher_dist = F.normalize(outputs_old["dist_features"], dim=-1)
            cosine_loss = 1 - (student_dist * teacher_dist).sum(dim=-1).mean()
            loss += self.lamb_cos * cosine_loss

        # --- Loss 3: CE with GT on current task (both heads) ---
        loss += F.cross_entropy(cls_logits[t], targets - self.model.task_offset[t])
        loss += F.cross_entropy(dist_logits[t], targets - self.model.task_offset[t])

        return loss
