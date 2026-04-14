import os
import warnings
import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader

from .incremental_learning import Incremental_Learning_Approach
from datasets.exemplars_dataset import ExemplarsDataset
from datasets.exemplars_selection import override_dataset_transform


class Appr(Incremental_Learning_Approach):
    """Incremental Classifier and Representation Learning (iCaRL)
    https://arxiv.org/abs/1611.07725

    Approach-specific args are read from args['approach_args']:
        lamb  (float, default 1.0) — distillation loss weight

    Exemplar args are read from args['exemplars_args'] by the trainer:
        num_exemplars           (int, default 0) — total exemplar budget
        num_exemplars_per_class (int, default 0) — per-class budget (mutually exclusive)
        exemplar_selection      (str, default 'random') — selection strategy
    """

    def __init__(self, args, model, logger=None, exemplars_dataset=None):
        super().__init__(args, model, logger, exemplars_dataset)
        self.model_old = None
        self.exemplar_means = []
        aargs = args.get('approach_args', {})
        self.lamb = aargs.get('lamb', 1.0)

        if self.exemplars_dataset is None:
            warnings.warn("Warning: iCaRL is expected to use exemplars. Check documentation.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    def _get_optimizer(self):
        """Returns the optimizer — iCaRL always trains all params (exemplars always present)."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)

    # Algorithm 2: iCaRL Incremental Train
    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # remove mean of exemplars during training — NCM classify not used during training
        self.exemplar_means = []

        # Algorithm 3: iCaRL Update Representation — form combined training set
        if t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        super().train_loop(t, trn_loader, val_loader)

        # Algorithm 4 & 5: ConstructExemplarSet + ReduceExemplarSet
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

        # compute mean of exemplars for NCM classification
        self.compute_mean_of_exemplars(trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Save a frozen copy of the model after each task — used as teacher for next task."""
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            outputs_old = None
            if t > 0:
                outputs_old = self.model_old(images.to(self.device))
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping)
            self.optimizer.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                outputs_old = None
                if t > 0:
                    outputs_old = self.model_old(images.to(self.device))
                outputs, feats = self.model(images.to(self.device), return_features=True)
                loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)
                # during training exemplar_means is empty — use standard logit-based metrics
                if not self.exemplar_means:
                    hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                else:
                    hits_taw, hits_tag = self.classify(t, feats, targets)
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    # Algorithm 3: classification and distillation terms
    def criterion(self, t, outputs, targets, outputs_old=None):
        """Returns the loss value"""
        # Classification loss over all heads
        loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        # Distillation loss for old classes — sigmoid BCE (original formulation)
        if t > 0:
            g = torch.sigmoid(torch.cat(outputs[:t], dim=1))
            q_i = torch.sigmoid(torch.cat(outputs_old[:t], dim=1))
            loss += self.lamb * sum(torch.nn.functional.binary_cross_entropy(g[:, y], q_i[:, y])
                                    for y in range(sum(self.model.task_cls[:t])))
        return loss

    # Algorithm 1: iCaRL NCM Classify
    def classify(self, task, features, targets):
        """Nearest-prototype classification using stored exemplar means."""
        means = torch.stack(self.exemplar_means)
        means = torch.stack([means] * features.shape[0])
        means = means.transpose(1, 2)
        features = features / features.norm(dim=1).view(-1, 1)
        features = features.unsqueeze(2)
        features = features.expand_as(means)
        dists = (features - means).pow(2).sum(1).squeeze()
        # Task-Aware
        num_cls = self.model.task_cls[task]
        offset = self.model.task_offset[task]
        pred = dists[:, offset:offset + num_cls].argmin(1)
        hits_taw = (pred + offset == targets.to(self.device)).float()
        # Task-Agnostic
        pred = dists.argmin(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def compute_mean_of_exemplars(self, trn_loader, transform):
        """Compute and store L2-normalized mean feature vector per exemplar class."""
        with override_dataset_transform(self.exemplars_dataset, transform) as _ds:
            icarl_loader = DataLoader(_ds, batch_size=trn_loader.batch_size, shuffle=False,
                                      num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
            extracted_features = []
            extracted_targets = []
            with torch.no_grad():
                self.model.eval()
                for images, targets in icarl_loader:
                    feats = self.model(images.to(self.device), return_features=True)[1]
                    extracted_features.append(feats / feats.norm(dim=1).view(-1, 1))
                    extracted_targets.extend(targets)
            extracted_features = torch.cat(extracted_features)
            extracted_targets = np.array(extracted_targets)
            for curr_cls in np.unique(extracted_targets):
                cls_ind = np.where(extracted_targets == curr_cls)[0]
                cls_feats = extracted_features[cls_ind]
                cls_feats_mean = cls_feats.mean(0) / cls_feats.mean(0).norm()
                self.exemplar_means.append(cls_feats_mean)

    def save_progress(self, results_path, task):
        """Save exemplars and exemplar means so resume works correctly."""
        if self.exemplars_dataset is not None:
            torch.save(
                {'images': self.exemplars_dataset.images, 'labels': self.exemplars_dataset.labels},
                os.path.join(results_path, f"task{task}_exemplars.pth")
            )
        if self.exemplar_means:
            torch.save(self.exemplar_means, os.path.join(results_path, f"task{task}_exemplar_means.pth"))

    def load_progress(self, results_path, task):
        """Restore model_old, exemplars, and exemplar means on resume."""
        # model_old: reconstructed from the checkpoint loaded by the trainer
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

        # exemplars
        exemplars_file = os.path.join(results_path, f"task{task}_exemplars.pth")
        if os.path.isfile(exemplars_file) and self.exemplars_dataset is not None:
            state = torch.load(exemplars_file, weights_only=False)
            self.exemplars_dataset.images = state['images']
            self.exemplars_dataset.labels = state['labels']
            print(f"Loaded {len(self.exemplars_dataset.images)} exemplars from {exemplars_file}")
        elif self.exemplars_dataset is not None:
            warnings.warn(f"Exemplars file NOT found at {exemplars_file}!")

        # exemplar means
        means_file = os.path.join(results_path, f"task{task}_exemplar_means.pth")
        if os.path.isfile(means_file):
            self.exemplar_means = torch.load(means_file, weights_only=False)
            print(f"Loaded {len(self.exemplar_means)} exemplar means from {means_file}")
        else:
            warnings.warn(f"Exemplar means file NOT found at {means_file}!")
