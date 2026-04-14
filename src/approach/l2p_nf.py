# NOTE: This approach has not yet been ported to the current framework interface and is non-functional.
# It is kept for reference. Do not use with the current trainer.

import torch
from argparse import ArgumentParser

from .inc_learn import Incremental_Learning_Approach
from datasets.exemplars_dataset import ExemplarsDataset
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

class Appr(Incremental_Learning_Approach):
    """Class implementing Learn to Prompt method"""

    def __init__(self,
                 model,
                 device,
                 nepochs=100,
                 lr_scheduler="none",
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
                 all_outputs=False):
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
        self.all_out = all_outputs

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--all-outputs', action='store_true', required=False,
                            help='Allow all weights related to all outputs to be modified (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1 and not self.all_out:
            # if there are no exemplars, previous heads are not modified
            params_all = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params_all = list(self.model.parameters())
        params = [p for p in params_all if p.requires_grad]
        trainable = sum(p.numel() for p in params)
        total = sum(p.numel() for p in params_all)

        print(f"[DEBUG] optimizer param numel -> total:{total} trainable:{trainable}")
        print(f"[DEBUG] optimizer tensors -> total:{len(list(params))} trainable:{sum(1 for p in params if p.requires_grad)}")
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
        return create_optimizer(args, model_without_ddp)
    
    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if self.all_out or len(self.exemplars_dataset) > 0:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
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