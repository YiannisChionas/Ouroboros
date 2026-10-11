"""ViT-Hydra — ViT with one cls token and N distillation tokens.

Generalization of DeiT (`VisionTransformerDistilled`, see deit_original.py) from
one distillation token to N. Each prefix token has its own linear head:

    token 0      (cls)    -> head          <- ground truth (CE)
    token 1..N   (dist_i) -> heads_dist[i] <- teacher i (soft KD)

Used for vit-hydra pretraining on ImageNet-1k (5 teachers, N=5).
"""
from typing import Optional

import torch
from torch import nn as nn

from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer, trunc_normal_

__all__ = ['VisionTransformerHydra', 'vit_hydra_base_patch16_224']


class VisionTransformerHydra(VisionTransformer):
    """Vision Transformer with N distillation tokens and N+1 heads."""

    def __init__(self, *args, num_dist_tokens=5, **kwargs):
        weight_init = kwargs.pop('weight_init', '')
        super().__init__(*args, **kwargs, weight_init='skip')
        assert self.global_pool in ('token',)
        assert not self.no_embed_class and self.reg_token is None

        self.num_dist_tokens = num_dist_tokens
        self.num_prefix_tokens = 1 + num_dist_tokens
        self.dist_tokens = nn.Parameter(torch.empty(1, num_dist_tokens, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.empty(1, self.patch_embed.num_patches + self.num_prefix_tokens, self.embed_dim))
        self.heads_dist = nn.ModuleList(
            [nn.Linear(self.embed_dim, self.num_classes) for _ in range(num_dist_tokens)])

        self.weight_init_mode = 'reset' if weight_init == 'skip' else weight_init
        if weight_init != 'skip':
            self.init_weights(needs_reset=False)

    def init_weights(self, mode='', needs_reset=True):
        mode = mode or self.weight_init_mode
        trunc_normal_(self.dist_tokens, std=.02)
        super().init_weights(mode=mode, needs_reset=needs_reset)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_tokens'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed|dist_tokens',
            blocks=[
                (r'^blocks\.(\d+)', None),
                (r'^norm', (99999,))]  # final norm w/ last block
        )

    def _pos_embed(self, x):
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=(H, W),
                old_size=self.patch_embed.grid_size,
                num_prefix_tokens=self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed
        # original timm / deit impl: pos_embed has entries for prefix tokens, concat then add
        x = torch.cat((
            self.cls_token.expand(x.shape[0], -1, -1),
            self.dist_tokens.expand(x.shape[0], -1, -1),
            x),
            dim=1)
        x = x + pos_embed
        return self.pos_drop(x)

    def forward_tokens(self, x):
        """Final-norm features of the prefix tokens: [B, 1 + N, D] (cls first)."""
        return self.forward_features(x)[:, :self.num_prefix_tokens]

    def forward_head(self, x, pre_logits: bool = False):
        """x: prefix token features [B, 1 + N, D] -> (cls_logits, [dist_logits_i])."""
        x = self.head_drop(x)
        if pre_logits:
            return x[:, 0], [x[:, i + 1] for i in range(self.num_dist_tokens)]
        cls_logits = self.head(x[:, 0])
        dist_logits = [h(x[:, i + 1]) for i, h in enumerate(self.heads_dist)]
        return cls_logits, dist_logits

    def forward(self, x):
        return self.forward_head(self.forward_tokens(x))


def vit_hydra_base_patch16_224(num_dist_tokens=5, num_classes=1000, drop_path_rate=0.1, **kwargs):
    """ViT-B/16 @224 (DeiT-B config) with `num_dist_tokens` distillation tokens, random init."""
    return VisionTransformerHydra(
        img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
        num_classes=num_classes, drop_path_rate=drop_path_rate,
        num_dist_tokens=num_dist_tokens, **kwargs)
