"""Learning to Prompt (L2P) for Class-Incremental Learning.

Wang et al., "Learning to Prompt for Continual Learning", CVPR 2022.
https://arxiv.org/abs/2112.08654

Architecture:
  - Frozen ViT backbone (vit_small_patch16_224_prompt or vit_base_patch16_224_prompt)
  - Learnable prompt pool (PromptPool inside ViT_Prompt)
  - Frozen reference backbone inside ViT_Prompt for query extraction (see vit_prompt.py)
  - Per-task linear heads

Loss = CE(outputs[t], local_targets) - lamb * reduce_sim
"""
import os
import torch
import torch.nn.functional as F

from .incremental_learning import Incremental_Learning_Approach


class Appr(Incremental_Learning_Approach):
    """L2P: prompt pool + heads trained; ViT backbone frozen."""

    def __init__(self, args, model, logger=None, exemplars_dataset=None):
        super().__init__(args, model, logger, exemplars_dataset)
        self.lamb = args.get('approach_args', {}).get('lamb', 0.5)

    # ------------------------------------------------------------------ optimizer

    def _get_optimizer(self):
        """Freeze ViT backbone, keep prompt pool + heads trainable."""
        # Freeze the full backbone (includes transformer weights)
        for p in self.model.model.parameters():
            p.requires_grad = False
        # Unfreeze only the prompt pool — backbone transformer stays frozen
        for p in self.model.model.prompt_pool.parameters():
            p.requires_grad = True

        params = list(self.model.model.prompt_pool.parameters()) + \
                 list(self.model.heads.parameters())
        return torch.optim.SGD(
            params, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum
        )

    # ------------------------------------------------------------------ loss

    def criterion(self, t, outputs, targets):
        """CE on the current task head (local labels) + prompt pull loss."""
        ce = F.cross_entropy(outputs[t], targets - self.model.task_offset[t])
        return ce - self.lamb * self.model.model.last_reduce_sim

    # ------------------------------------------------------------------ save / load

    def save_progress(self, results_path, task):
        path = os.path.join(results_path, f'prompt_pool_task{task}.pt')
        torch.save(self.model.model.prompt_pool.state_dict(), path)

    def load_progress(self, results_path, task):
        path = os.path.join(results_path, f'prompt_pool_task{task}.pt')
        self.model.model.prompt_pool.load_state_dict(
            torch.load(path, map_location=self.device)
        )
