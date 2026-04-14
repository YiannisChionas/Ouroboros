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
                 lamb=1):
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
                eval_outputs = outputs["cls_logits"]
                # eval_outputs = (outputs["cls_logits"] + outputs["dist_logits"]) / 2
                # eval_outputs = [
                #     (cls_out + dist_out) / 2
                #     for cls_out, dist_out in zip(outputs["cls_logits"], outputs["dist_logits"])
                # ]

                hits_taw, hits_tag = self.calculate_metrics(eval_outputs, targets)

                # Log
                total_loss += loss.data.cpu().numpy().item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def criterion(self, t, outputs, targets, outputs_old):
        loss = 0

        cls_logits = outputs["cls_logits"]
        dist_logits = outputs["dist_logits"]

        if t > 0 :
            # teacher_logits = (outputs_old["cls_logits"] + outputs_old["dist_logits"]) / 2
            teacher_logits = [
                (cls_out + dist_out) / 2
                for cls_out, dist_out in zip(outputs_old["cls_logits"], outputs_old["dist_logits"])
            ]

            student_dist  = torch.cat(dist_logits[:t], dim=1)
            teacher_out  = torch.cat(teacher_logits[:t], dim=1)

            teacher_targets = teacher_out.argmax(dim=1)

            # loss += self.lamb * torch.nn.functional.cross_entropy(input, target)
            loss += self.lamb * torch.nn.functional.cross_entropy(student_dist, teacher_targets)
        else:
            loss+= torch.nn.functional.cross_entropy(dist_logits[t], targets - self.model.task_offset[t])

        loss += torch.nn.functional.cross_entropy(cls_logits[t], targets - self.model.task_offset[t])

        return loss
