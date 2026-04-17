"""ViT with randomly-initialized distillation token (vit_dist).

Takes a pretrained ViT backbone and adds a dist_token initialized with
small random noise. Intended for ablation experiments comparing:
  - DeiT pretrained dist_token vs self-taught dist_token on ViT backbone
"""
import torch
from torch import nn

from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer, trunc_normal_
from timm.models._builder import build_model_with_cfg
from timm.models._registry import register_model

__all__ = ['VisionTransformerDistilledCIL']


class VisionTransformerDistilledCIL(VisionTransformer):
    """ViT-Small with a randomly-initialized distillation token.

    pos_embed covers patches only (no cls prefix position), matching the
    DeiT-CIL convention in FACILCUSTOM. Returns (cls_features, dist_features)
    as a tuple, compatible with LLL_Net_Distilled.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, weight_init='skip')
        assert self.global_pool in ('token',)

        dd = {'device': kwargs.get('device', None), 'dtype': kwargs.get('dtype', None)}

        self.dist_token = nn.Parameter(torch.empty(1, 1, self.embed_dim, **dd))
        # Override pos_embed: patch positions only (no cls prefix slot)
        self.pos_embed = nn.Parameter(torch.empty(1, self.patch_embed.num_patches, self.embed_dim, **dd))

        self.init_weights()

    def init_weights(self, mode='', needs_reset=True):
        mode = mode or self.weight_init_mode
        trunc_normal_(self.dist_token, std=.02)
        super().init_weights(mode=mode, needs_reset=needs_reset)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed|dist_token',
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    def _pos_embed(self, x):
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=(H, W),
                old_size=self.patch_embed.grid_size,
                num_prefix_tokens=0,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed
        # Add patch positional embeddings, then prepend cls and dist tokens
        x = x + pos_embed
        x = torch.cat((
            self.cls_token.expand(x.shape[0], -1, -1),
            self.dist_token.expand(x.shape[0], -1, -1),
            x,
        ), dim=1)
        return self.pos_drop(x)

    def forward_head(self, x, pre_logits: bool = False):
        # Returns tuple — matches LLL_Net_Distilled which does:
        #   cls_features, dist_features = outputs_backbone
        return x[:, 0], x[:, 1]


def checkpoint_filter_fn_vit_dist(state_dict, model):
    """Adapt a standard ViT checkpoint for VisionTransformerDistilledCIL.

    ViT pos_embed has shape [1, num_patches+1, D] (cls + patches).
    We strip the cls position and keep only patch positions, since
    cls and dist tokens are prepended without explicit pos embeddings.
    """
    out_dict = {}
    for k, v in state_dict.items():
        if k == 'pos_embed':
            # Strip cls position if present
            if v.ndim == 3 and v.shape[1] == model.patch_embed.num_patches + 1:
                v = v[:, 1:, :]
            # Resize if resolution mismatch
            if v.shape[1] != model.patch_embed.num_patches:
                old_size = int(v.shape[1] ** 0.5)
                v = resample_abs_pos_embed(
                    v,
                    new_size=model.patch_embed.grid_size,
                    old_size=(old_size, old_size),
                    num_prefix_tokens=0,
                )
            out_dict[k] = v
        else:
            out_dict[k] = v
    return out_dict


@register_model
def vit_small_patch16_224_dist(pretrained=False, **kwargs):
    """ViT-Small (augreg_in1k) backbone + randomly-initialized dist_token."""
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = build_model_with_cfg(
        VisionTransformerDistilledCIL,
        'vit_small_patch16_224.augreg_in1k',
        pretrained=pretrained,
        pretrained_filter_fn=checkpoint_filter_fn_vit_dist,
        **dict(model_args, **kwargs),
    )
    return model
