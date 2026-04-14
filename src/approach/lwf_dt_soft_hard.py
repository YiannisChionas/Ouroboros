import torch
from copy import deepcopy
from argparse import ArgumentParser

from .inc_learn import Incremental_Learning_Approach
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Incremental_Learning_Approach):
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
        self.T = T

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset
    
    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Page 5: "lambda is a loss balance weight, set to 1 for most our experiments. Making lambda larger will favor
        # the old task performance over the new task’s, so we can obtain a old-task-new-task performance line by
        # changing lambda."
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        for images, targets in trn_loader:
            # Forward old model
            outputs_old = None
            if t > 0:
                outputs_old = self.model_old(images.to(self.device))
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            # --- DEBUG INITIALIZATION ---
            stats = {i: {'cls_max': 0, 'dist_max': 0, 'cls_norm': 0, 'dist_norm': 0, 'count': 0} for i in range(t + 1)}
            # ----------------------------
            
            self.model.eval()
            for images, targets in val_loader:
                # Forward old model
                outputs_old = None
                if t > 0:
                    with torch.no_grad():
                        outputs_old = self.model_old(images.to(self.device))
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)

                # --- COLLECT DEBUG STATS ---
                for task_id in range(t + 1):
                    c_out = outputs["cls_logits"][task_id]
                    d_out = outputs["dist_logits"][task_id]
                    stats[task_id]['cls_max'] += c_out.max(dim=1)[0].sum().item()
                    stats[task_id]['dist_max'] += d_out.max(dim=1)[0].sum().item()
                    stats[task_id]['cls_norm'] += torch.norm(c_out, p=2, dim=1).sum().item()
                    stats[task_id]['dist_norm'] += torch.norm(d_out, p=2, dim=1).sum().item()
                    stats[task_id]['count'] += len(targets)
                # ---------------------------

                eval_outputs = outputs["cls_logits"]
                hits_taw, hits_tag = self.calculate_metrics(eval_outputs, targets)

                # Log
                total_loss += loss.data.cpu().numpy().item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)

            # --- PRINT DEBUG LOGGING AT THE END OF EVAL ---
            print("\n" + "="*40)
            print(f"DEBUG: Logit Statistics at Task {t}")
            for task_id in range(t + 1):
                s = stats[task_id]
                c = s['count']
                print(f"Task {task_id}:")
                print(f"  CLS  -> AvgMax: {s['cls_max']/c:.2f}, AvgNorm: {s['cls_norm']/c:.2f}")
                print(f"  DIST -> AvgMax: {s['dist_max']/c:.2f}, AvgNorm: {s['dist_norm']/c:.2f}")
            print("="*40 + "\n")

        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
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

        if t > 0 :
            teacher_old = torch.cat(outputs_old["cls_logits"][:t], dim=1)

            # Soft Knowledge distillation for cls
            loss += self.lamb * self.cross_entropy(
                torch.cat(cls_logits[:t], dim=1),
                torch.cat(outputs_old["cls_logits"][:t],dim=1),
                exp=1.0 / self.T)

            # Hard Knowledge distillation for dist
            teacher_targets = teacher_old.argmax(dim=1)
            loss += self.lamb * torch.nn.functional.cross_entropy(torch.cat(dist_logits[:t], dim=1),teacher_targets)

        loss += torch.nn.functional.cross_entropy(cls_logits[t], targets - self.model.task_offset[t])

        return loss
