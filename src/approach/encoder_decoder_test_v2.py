import os
import torch
import torch.nn.functional as F
from copy import deepcopy

from .incremental_learning import Incremental_Learning_Approach
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Incremental_Learning_Approach):
    """Encoder-decoder task retrieval with orthogonality loss (v2).

    Training:
      - cls head, old tasks : soft KL distillation on cls logits
      - dist token, old tasks: cosine similarity loss on backbone features (local stability)
      - dist token, old tasks: orthogonality loss vs stored prototypes (global separability)
      - current task        : CE with GT on cls head only

    After each task: store mean dist_features as prototype.

    Inference:
      - dist_features → cosine similarity with all prototypes → predicted task
      - cls_logits[predicted_task] → final class prediction (retrieval-based TAg)

    Logged metrics:
      - acc_tag_std   : standard TAg (argmax over all heads, no retrieval)
      - retrieval_acc : fraction of samples where task retrieval is correct
    Returned acc_tag  : retrieval-based TAg
    """

    def __init__(self, args, model, logger=None, exemplars_dataset=None):
        super().__init__(args, model, logger, exemplars_dataset)
        aargs = args.get('approach_args', {})
        self.lamb      = aargs.get('lamb', 1.0)
        self.lamb_cos  = aargs.get('lamb_cos', 1.0)
        self.lamb_orth = aargs.get('lamb_orth', 1.0)
        self.T         = aargs.get('T', 2)

        self.model_old  = None
        self.prototypes = []  # list of [D] cpu tensors, one per task

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def post_train_process(self, t, trn_loader):
        # Save teacher
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

        # Compute and store prototype for task t
        self.model.eval()
        feats = []
        with torch.no_grad():
            for images, _ in trn_loader:
                out = self.model(images.to(self.device), return_features=True)
                feats.append(out['dist_features'])
        prototype = torch.cat(feats, dim=0).mean(0).cpu()  # [D]
        self.prototypes.append(prototype)
        print(f'Prototype stored for task {t} — total: {len(self.prototypes)}')

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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping)
            self.optimizer.step()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval(self, t, val_loader):
        total_loss, total_taw, total_tag_std, total_tag_ret, total_ret_acc, total_num = \
            0.0, 0, 0, 0, 0, 0

        proto_mat = None
        if len(self.prototypes) > 1:
            proto_mat = F.normalize(
                torch.stack(self.prototypes).to(self.device), dim=-1
            )  # [T_so_far, D]

        self.model.eval()
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)

                outputs_old = None
                if t > 0:
                    outputs_old = self.model_old(images, return_features=True)
                outputs = self.model(images, return_features=True)

                loss       = self.criterion(t, outputs, targets.to(self.device), outputs_old)
                cls_logits = outputs['cls_logits']

                hits_taw, hits_tag_std = self.calculate_metrics(cls_logits, targets)

                if proto_mat is not None:
                    dist_feat = F.normalize(outputs['dist_features'], dim=-1)  # [B, D]
                    sims      = dist_feat @ proto_mat.T                         # [B, T]
                    pred_task = sims.argmax(dim=1)                              # [B]

                    hits_tag_ret = self._retrieval_hits(cls_logits, targets.to(self.device), pred_task)
                    hits_ret_acc = (pred_task == t).sum().item()
                else:
                    hits_tag_ret = hits_tag_std.sum().item()
                    hits_ret_acc = len(targets)

                total_loss    += loss.item() * len(targets)
                total_taw     += hits_taw.sum().item()
                total_tag_std += hits_tag_std.sum().item()
                total_tag_ret += hits_tag_ret
                total_ret_acc += hits_ret_acc
                total_num     += len(targets)

        self.logger.log_scalar(task=t, iter=t, name='retrieval_acc', value=total_ret_acc / total_num, group='test')
        self.logger.log_scalar(task=t, iter=t, name='acc_tag_std',   value=total_tag_std / total_num, group='test')

        return total_loss / total_num, total_taw / total_num, total_tag_ret / total_num

    def _retrieval_hits(self, cls_logits, targets, pred_task):
        """Per-sample: use cls_logits[pred_task[i]] + task offset for global class prediction."""
        logits_stack = torch.stack(cls_logits, dim=1)                     # [B, T, n_cls_per_task]
        B, _, n      = logits_stack.shape
        idx          = pred_task.view(B, 1, 1).expand(B, 1, n)
        selected     = logits_stack.gather(1, idx).squeeze(1)             # [B, n_cls]
        task_off     = self.model.task_offset.to(self.device)[pred_task]  # [B]
        pred_cls     = selected.argmax(dim=1) + task_off                  # [B] global class
        return (pred_cls == targets).sum().item()

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

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

    def criterion(self, t, outputs, targets, outputs_old):
        loss       = 0
        cls_logits = outputs['cls_logits']

        if t > 0:
            # KL distillation on cls logits
            loss += self.lamb * self.cross_entropy(
                torch.cat(cls_logits[:t], dim=1),
                torch.cat(outputs_old['cls_logits'][:t], dim=1),
                exp=1.0 / self.T
            )

            # Cosine loss on dist features (local stability)
            student_dist = F.normalize(outputs['dist_features'],     dim=-1)
            teacher_dist = F.normalize(outputs_old['dist_features'], dim=-1)
            loss += self.lamb_cos * (1 - (student_dist * teacher_dist).sum(dim=-1).mean())

            # Orthogonality loss vs old task prototypes (global separability)
            proto_stack = F.normalize(
                torch.stack(self.prototypes).to(self.device), dim=-1
            )  # [t, D]
            # cosine similarity between current dist_features and each old prototype → push to 0
            orth_loss = (student_dist @ proto_stack.T).abs().mean()
            loss += self.lamb_orth * orth_loss

        loss += F.cross_entropy(cls_logits[t], targets - self.model.task_offset[t])
        return loss

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def save_progress(self, results_path, task):
        torch.save(self.prototypes, os.path.join(results_path, 'prototypes.pth'))

    def load_progress(self, results_path, task):
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

        pt_file = os.path.join(results_path, 'prototypes.pth')
        if os.path.isfile(pt_file):
            self.prototypes = torch.load(pt_file, weights_only=False)
            print(f'Loaded {len(self.prototypes)} prototypes from {pt_file}')
