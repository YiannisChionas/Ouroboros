import torch

from .incremental_learning_v3 import Incremental_Learning_Approach
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Incremental_Learning_Approach):
    """Class implementing the finetuning baseline

    Approach-specific args are read from args['approach_args']:
        all_outputs  (bool, default False) — use all heads for loss (needed with exemplars)
    """

    def __init__(self, args, model, logger=None, exemplars_dataset=None):
        super().__init__(args, model, logger, exemplars_dataset)
        aargs = args.get('approach_args', {})
        self.all_out = aargs.get('all_outputs', False)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if not self.exemplars_dataset and len(self.model.heads) > 1 and not self.all_out:
            # if there are no exemplars, previous heads are not modified
            params_all = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params_all = list(self.model.parameters())
        params = [p for p in params_all if p.requires_grad]
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if self.exemplars_dataset and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        super().train_loop(t, trn_loader, val_loader)

        if self.exemplars_dataset:
            self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if self.all_out or self.exemplars_dataset:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
