"""Learning to Prompt (L2P) for Class-Incremental Learning.

Wang et al., "Learning to Prompt for Continual Learning", CVPR 2022.
https://arxiv.org/abs/2112.08654

Architecture:
  - Frozen ViT backbone (vit_small_patch16_224_prompt)
  - Learnable prompt pool (PromptPool inside ViT_Prompt)
  - Per-task linear heads (same as LwF / finetuning)

Loss = CE(concat_heads, targets) - lamb * reduce_sim
where reduce_sim is the mean query-key cosine similarity stored by ViT_Prompt.
"""
import os
import torch
import torch.nn.functional as F

from .incremental_learning import Incremental_Learning_Approach
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Incremental_Learning_Approach):
    """L2P approach — prompt pool trained, backbone frozen."""

    def __init__(self, args, model, logger=None, exemplars_dataset=None):
        super().__init__(args, model, logger, exemplars_dataset)
        aargs = args.get('approach_args', {})
        self.lamb = aargs.get('lamb', 0.5)   # weight for reduce_sim pull loss

    # ------------------------------------------------------------------ setup

    def _get_optimizer(self):
        """Only prompt pool + current head are trained; backbone stays frozen."""
        prompt_params = list(self.model.model.prompt_pool.parameters())
        head_params   = list(self.model.heads[-1].parameters())
        params = [p for p in prompt_params + head_params if p.requires_grad]
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)

    # ------------------------------------------------------------------ train

    def train_epoch(self, t, trn_loader):
        self.model.train()
        for images, targets in trn_loader:
            images  = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)
            loss    = self.criterion(t, outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            # clip only trainable params
            trainable = list(self.model.model.prompt_pool.parameters()) + \
                        list(self.model.heads[-1].parameters())
            torch.nn.utils.clip_grad_norm_(trainable, self.clipping)
            self.optimizer.step()

    def criterion(self, t, outputs, targets):
        """CE on all seen heads (flat) + pull loss on prompt similarity."""
        ce = F.cross_entropy(torch.cat(outputs[:t + 1], dim=1), targets)
        reduce_sim = self.model.model.last_reduce_sim
        return ce - self.lamb * reduce_sim

    # ------------------------------------------------------------------ save / load

    def save_progress(self, results_path, task):
        """Persist prompt pool so training can be resumed."""
        path = os.path.join(results_path, f'prompt_pool_task{task}.pt')
        torch.save(self.model.model.prompt_pool.state_dict(), path)

    def load_progress(self, results_path, task):
        """Restore prompt pool from a previous run."""
        path = os.path.join(results_path, f'prompt_pool_task{task}.pt')
        state = torch.load(path, map_location=self.device)
        self.model.model.prompt_pool.load_state_dict(state)
