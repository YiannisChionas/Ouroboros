"""Learning a Unified Classifier Incrementally via Rebalancing (LUCIR).

Hou et al., CVPR 2019.
https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.pdf
Original code: https://github.com/hshustc/CVPR19_Incremental_Learning

Core ideas:
  - CosineLinear heads: cosine-normalized classifier (features and weights L2-normalised)
    scaled by a learnable sigma (eta). Removes the magnitude bias between old/new heads.
  - Less-Forgetting constraint (default on): penalises change in feature space via
    CosineEmbeddingLoss between current and frozen-teacher features.
  - Inter-Class Separation (margin ranking, default on): pushes old-class scores above
    the top-K new-class scores for samples from old classes.
  - Adaptive lambda: scales the distillation weight by sqrt(#old / #new) so the balance
    is maintained as the number of seen classes grows.

Approach-specific args (args['approach_args']):
    lamb              (float, default 5.0)  — base distillation weight
    lamb_mr           (float, default 1.0)  — margin ranking weight
    dist              (float, default 0.5)  — margin threshold for ranking loss
    K                 (int,   default 2)    — hard negatives per sample for ranking loss
    remove_less_forget      (bool, default False) — disable Less-Forgetting constraint
    remove_margin_ranking   (bool, default False) — disable Inter-Class Separation loss
    remove_adapt_lamda      (bool, default False) — disable adaptive lambda scaling
"""
import os
import math
import warnings
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .incremental_learning import Incremental_Learning_Approach
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Incremental_Learning_Approach):

    def __init__(self, args, model, logger=None, exemplars_dataset=None):
        super().__init__(args, model, logger, exemplars_dataset)
        aargs = args.get('approach_args', {})
        self.lamb     = aargs.get('lamb',   5.0)
        self.lamb_mr  = aargs.get('lamb_mr', 1.0)
        self.dist     = aargs.get('dist',   0.5)
        self.K        = aargs.get('K',      2)
        self.less_forget    = not aargs.get('remove_less_forget',    False)
        self.margin_ranking = not aargs.get('remove_margin_ranking', False)
        self.adapt_lamda    = not aargs.get('remove_adapt_lamda',    False)

        self.lamda    = self.lamb   # effective lambda, updated per task if adapt_lamda
        self.ref_model = None

        # Warmup loss must handle CosineLinear dict outputs during training
        self.warmup_loss = self._warmup_lucir_loss

        if self.exemplars_dataset is None:
            warnings.warn("LUCIR is expected to use exemplars. Check documentation.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    # ------------------------------------------------------------------ optimizer

    def _get_optimizer(self):
        if self.less_forget:
            # Old heads frozen — only backbone + new head updated
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = list(self.model.parameters())
        params = [p for p in params if p.requires_grad]
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)

    # ------------------------------------------------------------------ task hooks

    def pre_train_process(self, t, trn_loader):
        """Replace new head with CosineLinear; share sigma; freeze old heads if less_forget."""
        self.model.heads[-1] = CosineLinear(
            self.model.heads[-1].in_features,
            self.model.heads[-1].out_features,
        ).to(self.device)

        if t > 0:
            # Share the single sigma scalar across all heads (Eq. 4 in paper)
            self.model.heads[-1].sigma = self.model.heads[-2].sigma

            if self.less_forget:
                for h in self.model.heads[:-1]:
                    for param in h.parameters():
                        param.requires_grad = False
                self.model.heads[-1].sigma.requires_grad = True

            if self.adapt_lamda:
                n_old = sum(h.out_features for h in self.model.heads[:-1])
                n_new = self.model.heads[-1].out_features
                self.lamda = self.lamb * math.sqrt(n_old / n_new)

        super().pre_train_process(t, trn_loader)

    def train_loop(self, t, trn_loader, val_loader):
        if self.exemplars_dataset is not None and t > 0:
            trn_loader = torch.utils.data.DataLoader(
                trn_loader.dataset + self.exemplars_dataset,
                batch_size=trn_loader.batch_size, shuffle=True,
                num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory,
            )

        super().train_loop(t, trn_loader, val_loader)

        if self.exemplars_dataset is not None:
            self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Freeze a copy of the model as reference teacher for the next task."""
        self.ref_model = deepcopy(self.model)
        self.ref_model.eval()
        # Heads in train mode so CosineLinear returns {'wsigma', 'wosigma'} — needed in criterion
        for h in self.ref_model.heads:
            h.train()
        self.ref_model.freeze_all()

    # ------------------------------------------------------------------ train / eval

    def train_epoch(self, t, trn_loader):
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            images, targets = images.to(self.device), targets.to(self.device)
            outputs, features = self.model(images, return_features=True)
            ref_outputs, ref_features = None, None
            if t > 0:
                with torch.no_grad():
                    ref_outputs, ref_features = self.ref_model(images, return_features=True)
            loss = self.criterion(t, outputs, targets, ref_outputs, features, ref_features)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping)
            self.optimizer.step()

    def eval(self, t, val_loader):
        total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
        self.model.eval()
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs, features = self.model(images, return_features=True)
                ref_outputs, ref_features = None, None
                if t > 0:
                    ref_outputs, ref_features = self.ref_model(images, return_features=True)
                loss = self.criterion(t, outputs, targets, ref_outputs, features, ref_features)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                total_loss    += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num     += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    # ------------------------------------------------------------------ loss

    def criterion(self, t, outputs, targets, ref_outputs=None, features=None, ref_features=None):
        # Task 0: plain CE on CosineLinear output (wsigma during training, plain tensor during eval)
        if t == 0 or ref_outputs is None:
            out = torch.cat([o['wsigma'] if isinstance(o, dict) else o for o in outputs], dim=1)
            return F.cross_entropy(out, targets)

        # --- Less-Forgetting or MSE distillation ---
        if self.less_forget:
            # Eq. 6: penalise angular drift in feature space
            loss_dist = nn.CosineEmbeddingLoss()(
                features, ref_features.detach(),
                torch.ones(targets.shape[0], device=self.device),
            ) * self.lamda
        else:
            # Eq. 5: MSE on pre-sigma cosine scores for old classes
            ref_scores = torch.cat([ro['wosigma'] for ro in ref_outputs[:t]], dim=1).detach()
            old_scores = torch.cat([o['wosigma']  for o  in outputs[:t]],    dim=1)
            loss_dist  = nn.MSELoss()(old_scores, ref_scores) * self.lamda * ref_scores.shape[1]

        # --- Inter-Class Separation (margin ranking) ---
        loss_mr = torch.zeros(1, device=self.device)
        if self.margin_ranking:
            outputs_wos    = torch.cat([o['wosigma'] for o in outputs], dim=1)
            num_old        = outputs_wos.shape[1] - outputs[-1]['wosigma'].shape[1]
            hard_mask      = targets < num_old
            hard_num       = hard_mask.sum()
            if hard_num > 0:
                gt_scores        = outputs_wos.gather(1, targets[hard_mask].unsqueeze(1)).repeat(1, self.K)
                max_novel_scores = outputs_wos[hard_mask, num_old:].topk(self.K, dim=1)[0]
                loss_mr = nn.MarginRankingLoss(margin=self.dist)(
                    gt_scores.reshape(-1, 1),
                    max_novel_scores.reshape(-1, 1),
                    torch.ones(hard_num * self.K, device=self.device),
                ) * self.lamb_mr

        # --- CE ---
        out_wsigma = torch.cat([o['wsigma'] for o in outputs], dim=1)
        loss_ce = F.cross_entropy(out_wsigma, targets)

        return loss_dist + loss_ce + loss_mr

    # ------------------------------------------------------------------ warmup

    @staticmethod
    def _warmup_lucir_loss(outputs, targets):
        """CE loss for the head warmup phase; handles CosineLinear dict output."""
        o = outputs['wosigma'] if isinstance(outputs, dict) else outputs
        return F.cross_entropy(o, targets)

    # ------------------------------------------------------------------ resume

    def save_progress(self, results_path, task):
        if self.exemplars_dataset is not None:
            torch.save(
                {'images': self.exemplars_dataset.images, 'labels': self.exemplars_dataset.labels},
                os.path.join(results_path, f"task{task}_exemplars.pth"),
            )

    def load_progress(self, results_path, task):
        # Reconstruct ref_model from the already-loaded checkpoint
        self.ref_model = deepcopy(self.model)
        self.ref_model.eval()
        for h in self.ref_model.heads:
            h.train()
        self.ref_model.freeze_all()

        ex_file = os.path.join(results_path, f"task{task}_exemplars.pth")
        if os.path.isfile(ex_file) and self.exemplars_dataset is not None:
            state = torch.load(ex_file, weights_only=False)
            self.exemplars_dataset.images = state['images']
            self.exemplars_dataset.labels = state['labels']
            print(f"Loaded {len(self.exemplars_dataset.images)} exemplars from {ex_file}")
        elif self.exemplars_dataset is not None:
            warnings.warn(f"Exemplars file not found at {ex_file}!")


# ------------------------------------------------------------------ CosineLinear

class CosineLinear(nn.Module):
    """Cosine-normalised linear layer with learnable scale sigma (eta in paper).

    During training returns {'wsigma': sigma*cos, 'wosigma': cos}.
    During eval returns sigma*cos directly (compatible with calculate_metrics).
    """

    def __init__(self, in_features, out_features, sigma=True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight       = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma        = nn.Parameter(torch.ones(1)) if sigma else None
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        out = F.linear(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        out_s = self.sigma * out if self.sigma is not None else out
        if self.training:
            return {'wsigma': out_s, 'wosigma': out}
        return out_s
