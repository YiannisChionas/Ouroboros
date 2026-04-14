import time
import warnings
import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader

from .incremental_learning import Incremental_Learning_Approach
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Incremental_Learning_Approach):
    """Bias Correction (BiC)
    http://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.pdf

    Three-stage training per task:
      Stage 1 — LwF-style distillation + exemplars (backbone training)
      Stage 2 — Bias correction: learn alpha/beta per task head on held-out val exemplars
      Stage 3 — Exemplar management

    Approach-specific args are read from args['approach_args']:
        lamb                   (float, default -1)   — KD weight; -1 = n/(n+m) auto schedule
        T                      (int,   default 2)     — distillation temperature
        val_exemplar_percentage (float, default 0.1)  — fraction of exemplars reserved for bias correction val set
        num_bias_epochs        (int,   default 200)   — epochs for Stage 2 bias training

    Exemplar args are read from args['exemplars_args'] by the trainer:
        num_exemplars           (int, default 0) — total exemplar budget
        num_exemplars_per_class (int, default 0) — per-class budget (mutually exclusive)
        exemplar_selection      (str, default 'random') — selection strategy
    """

    def __init__(self, args, model, logger=None, exemplars_dataset=None):
        super().__init__(args, model, logger, exemplars_dataset)
        aargs = args.get('approach_args', {})
        self.lamb        = aargs.get('lamb', -1)
        self.T           = aargs.get('T', 2)
        self.val_pct     = aargs.get('val_exemplar_percentage', 0.1)
        self.bias_epochs = aargs.get('num_bias_epochs', 200)

        self.model_old   = None
        self.bias_layers = []

        # Validation exemplar store (split off from the training exemplar budget)
        self.x_valid_exemplars = []
        self.y_valid_exemplars = []

        if self.exemplars_dataset is None:
            warnings.warn("BiC requires exemplars. Check exemplars_args in config.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    # ------------------------------------------------------------------
    # Training loop — 3 stages
    # ------------------------------------------------------------------

    def train_loop(self, t, trn_loader, val_loader):

        # Add a bias layer for the new task
        self.bias_layers.append(BiasLayer().to(self.device))

        # Cache original exemplar budget before we temporarily modify it
        orig_max_exemplars       = self.exemplars_dataset.max_num_exemplars
        orig_max_exemplars_pcls  = self.exemplars_dataset.max_num_exemplars_per_class

        # ----------------------------------------------------------------
        # STAGE 0: Split exemplar budget into train / validation subsets
        # ----------------------------------------------------------------
        print('Stage 0: Select validation exemplars for bias correction')
        clock0 = time.time()

        num_cls     = sum(self.model.task_cls)
        num_old_cls = sum(self.model.task_cls[:t])

        if orig_max_exemplars != 0:
            num_ex_pcls     = int(np.floor(orig_max_exemplars / num_cls))
            num_val_ex_pcls = int(np.ceil(self.val_pct * num_ex_pcls))
            num_trn_ex_pcls = num_ex_pcls - num_val_ex_pcls
            self.exemplars_dataset.max_num_exemplars = (num_trn_ex_pcls * num_cls).item()
        else:
            num_val_ex_pcls = int(np.ceil(self.val_pct * orig_max_exemplars_pcls))
            num_trn_ex_pcls = orig_max_exemplars_pcls - num_val_ex_pcls
            self.exemplars_dataset.max_num_exemplars_per_class = num_trn_ex_pcls

        # Trim stored val exemplars for old classes to match new budget
        if t > 0:
            if orig_max_exemplars != 0:
                num_ex_pcls_old     = int(np.floor(orig_max_exemplars / num_old_cls))
                num_val_ex_pcls_old = int(np.ceil(self.val_pct * num_ex_pcls_old))
                for cls in range(num_old_cls):
                    self.x_valid_exemplars[cls] = self.x_valid_exemplars[cls][:num_val_ex_pcls]
                    self.y_valid_exemplars[cls] = self.y_valid_exemplars[cls][:num_val_ex_pcls]

        # Collect val exemplars for new classes from trn_loader
        non_selected = []
        for curr_cls in range(num_old_cls, num_cls):
            self.x_valid_exemplars.append([])
            self.y_valid_exemplars.append([])
            cls_ind = np.where(np.asarray(trn_loader.dataset.labels) == curr_cls)[0]
            assert len(cls_ind) > 0, f"No samples for class {curr_cls}"
            assert num_val_ex_pcls <= len(cls_ind), f"Not enough samples for class {curr_cls}"
            self.x_valid_exemplars[curr_cls] = [trn_loader.dataset.images[i] for i in cls_ind[:num_val_ex_pcls]]
            self.y_valid_exemplars[curr_cls] = [trn_loader.dataset.labels[i] for i in cls_ind[:num_val_ex_pcls]]
            non_selected.extend(cls_ind[num_val_ex_pcls:])

        # Remove val samples from the training set
        trn_loader.dataset.images = [trn_loader.dataset.images[i] for i in non_selected]
        trn_loader.dataset.labels = [trn_loader.dataset.labels[i] for i in non_selected]

        clock1 = time.time()
        n_val = sum(len(e) for e in self.y_valid_exemplars)
        print(f' > Selected {n_val} val exemplars, time={clock1 - clock0:.1f}s')

        # Keep a copy of the current task dataset structure for Stage 2 loader
        bic_val_dataset = deepcopy(trn_loader.dataset)

        # Merge with stored exemplars for Stage 1 training
        if t > 0:
            trn_loader = DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                    batch_size=trn_loader.batch_size, shuffle=True,
                                    num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)

        # ----------------------------------------------------------------
        # STAGE 1: Backbone training with LwF distillation
        # ----------------------------------------------------------------
        print('Stage 1: Backbone training with distillation')
        super().train_loop(t, trn_loader, val_loader)

        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

        # ----------------------------------------------------------------
        # STAGE 2: Bias correction layer training
        # ----------------------------------------------------------------
        if t > 0:
            print('Stage 2: Training bias correction layers')

            if isinstance(bic_val_dataset.images, list):
                bic_val_dataset.images = sum(self.x_valid_exemplars, [])
            else:
                bic_val_dataset.images = np.vstack(self.x_valid_exemplars)
            bic_val_dataset.labels = sum(self.y_valid_exemplars, [])

            bic_val_loader = DataLoader(bic_val_dataset, batch_size=trn_loader.batch_size,
                                        shuffle=True, num_workers=trn_loader.num_workers,
                                        pin_memory=trn_loader.pin_memory)

            self.model.eval()
            self.bias_layers[t].alpha.requires_grad = True
            self.bias_layers[t].beta.requires_grad = True

            bic_optimizer = torch.optim.SGD(self.bias_layers[t].parameters(), lr=self.lr, momentum=0.9)

            for e in range(self.bias_epochs):
                clock0 = time.time()
                total_loss, total_acc = 0.0, 0.0
                for inputs, targets in bic_val_loader:
                    inputs  = inputs.to(self.device)
                    targets = targets.to(self.device)
                    with torch.no_grad():
                        outputs     = self.model(inputs)
                        old_outputs = self.bias_forward(outputs[:t])
                    new_output  = self.bias_layers[t](outputs[t])
                    pred_all    = torch.cat([torch.cat(old_outputs, dim=1), new_output], dim=1)
                    loss = torch.nn.functional.cross_entropy(pred_all, targets)
                    loss += 0.1 * (self.bias_layers[t].beta[0] ** 2) / 2
                    total_loss += loss.item() * len(targets)
                    total_acc  += (pred_all.argmax(1) == targets).float().sum().item()
                    bic_optimizer.zero_grad()
                    loss.backward()
                    bic_optimizer.step()
                clock1 = time.time()
                if (e + 1) % max(1, self.bias_epochs // 4) == 0:
                    n = len(bic_val_loader.dataset.labels)
                    print(f'| Epoch {e+1:3d}, time={clock1-clock0:.1f}s'
                          f' | Train: loss={total_loss/n:.3f}, TAg acc={100*total_acc/n:5.1f}% |')

            self.bias_layers[t].alpha.requires_grad = False
            self.bias_layers[t].beta.requires_grad = False

        for task in range(t + 1):
            print(f'Stage 2: BiC Task {task}: alpha={self.bias_layers[task].alpha.item():.5f}'
                  f', beta={self.bias_layers[task].beta.item():.5f}')

        # ----------------------------------------------------------------
        # STAGE 3: Exemplar management
        # ----------------------------------------------------------------
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    # ------------------------------------------------------------------
    # Per-epoch training
    # ------------------------------------------------------------------

    def train_epoch(self, t, trn_loader):
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            targets_old = None
            if t > 0:
                with torch.no_grad():
                    targets_old = self.bias_forward(self.model_old(images.to(self.device)))
            outputs = self.bias_forward(self.model(images.to(self.device)))
            loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping)
            self.optimizer.step()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval(self, t, val_loader):
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0.0, 0.0, 0.0, 0
            self.model.eval()
            for images, targets in val_loader:
                targets_old = None
                if t > 0:
                    targets_old = self.bias_forward(self.model_old(images.to(self.device)))
                outputs = self.bias_forward(self.model(images.to(self.device)))
                loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                total_loss    += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num     += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def criterion(self, t, outputs, targets, targets_old=None):
        loss_dist = 0.0
        if t > 0:
            loss_dist = self._soft_cross_entropy(
                torch.cat(outputs[:t],     dim=1),
                torch.cat(targets_old[:t], dim=1),
                T=self.T,
            )
        if self.lamb == -1:
            lamb = (self.model.task_cls[:t].sum().float() / self.model.task_cls.sum()).to(self.device)
        else:
            lamb = self.lamb
        ce = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return (1.0 - lamb) * ce + lamb * loss_dist

    @staticmethod
    def _soft_cross_entropy(outputs, targets, T=2, eps=1e-5):
        """Soft cross-entropy with temperature scaling (KD loss)."""
        out = torch.nn.functional.softmax(outputs / T, dim=1)
        tar = torch.nn.functional.softmax(targets / T, dim=1)
        out = (out + eps / out.size(1)) / (1 + eps)
        return -(tar * out.log()).sum(1).mean()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def bias_forward(self, outputs):
        """Apply per-task bias layers to a list of head outputs."""
        return [self.bias_layers[m](outputs[m]) for m in range(len(outputs))]

    def save_progress(self, results_path, task):
        """Save bias layers and val exemplars so resume works."""
        import os
        torch.save([bl.state_dict() for bl in self.bias_layers],
                   os.path.join(results_path, f"task{task}_bias_layers.pth"))
        torch.save({'x': self.x_valid_exemplars, 'y': self.y_valid_exemplars},
                   os.path.join(results_path, f"task{task}_val_exemplars.pth"))

    def load_progress(self, results_path, task):
        """Restore model_old, bias layers and val exemplars on resume."""
        import os, warnings
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

        bias_file = os.path.join(results_path, f"task{task}_bias_layers.pth")
        if os.path.isfile(bias_file):
            states = torch.load(bias_file, weights_only=False)
            self.bias_layers = []
            for sd in states:
                bl = BiasLayer().to(self.device)
                bl.load_state_dict(sd)
                self.bias_layers.append(bl)
            print(f"Loaded {len(self.bias_layers)} bias layers from {bias_file}")
        else:
            warnings.warn(f"Bias layers file NOT found at {bias_file}!")

        val_file = os.path.join(results_path, f"task{task}_val_exemplars.pth")
        if os.path.isfile(val_file):
            state = torch.load(val_file, weights_only=False)
            self.x_valid_exemplars = state['x']
            self.y_valid_exemplars = state['y']
            print(f"Loaded val exemplars from {val_file}")
        else:
            warnings.warn(f"Val exemplars file NOT found at {val_file}!")


# ------------------------------------------------------------------

class BiasLayer(torch.nn.Module):
    """Per-task affine correction: alpha * logits + beta."""

    def __init__(self):
        super().__init__()
        # Initialized on CPU; caller moves to device via .to(device)
        self.alpha = torch.nn.Parameter(torch.ones(1),  requires_grad=False)
        self.beta  = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        return self.alpha * x + self.beta
