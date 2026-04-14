from functools import partial
from typing import Optional, Type

import torch
from torch import nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer, trunc_normal_, checkpoint_filter_fn
from timm.models._builder import build_model_with_cfg
from timm.models._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = ['VisionTransformerDistilledCIL']  # model_registry will add each entrypoint fn to this


class VisionTransformerDistilledCIL(VisionTransformer):
    def __init__(self, *args,  distilled: bool = False,**kwargs):
        super().__init__(*args, **kwargs, weight_init='skip')
        
        assert self.global_pool in ('token',)

        dd = {'device': kwargs.get('device', None), 'dtype': kwargs.get('dtype', None)}

        self.is_distilled = distilled

        self.dist_token = nn.Parameter(torch.empty(1, 1, self.embed_dim, **dd))
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
            blocks=[
                (r'^blocks\.(\d+)', None),
                (r'^norm', (99999,))]  # final norm w/ last block
        )
    
    # Adapt positional embed to comply with distilation head
    def _pos_embed(self, x):
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            prev_grid_size = self.patch_embed.grid_size
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=(H, W),
                old_size=prev_grid_size,
                num_prefix_tokens=0,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed
        x = x + pos_embed 
        tokens = [self.cls_token.expand(x.shape[0], -1, -1)]
        if self.is_distilled:
            tokens.append(self.dist_token.expand(x.shape[0], -1, -1))
        x = torch.cat((*tokens, x), dim=1)
        return self.pos_drop(x)

    def forward_head(self, x, pre_logits: bool = False):
        cls_feat = x[:, 0]
        dist_feat = x[:, 1] if self.is_distilled else None
        return {
            "cls_features": cls_feat,
            "dist_features": dist_feat,
        }

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': .9,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        'license': 'apache-2.0',
        **kwargs
    }


def checkpoint_filter_fn_cil(state_dict, model, interpolation='bicubic', antialias=True):
    out_dict = {}

    for k, v in state_dict.items():
        if k == 'pos_embed':

            if v.ndim == 3 and v.shape[1] == model.patch_embed.num_patches + 1:
                v = v[:, 1:, :]   # keep only patch positional embeddings

            if v.shape[1] != model.patch_embed.num_patches:
                old_size = int(v.shape[1] ** 0.5)
                new_size = model.patch_embed.grid_size
                v = resample_abs_pos_embed(
                    v,
                    new_size=new_size,
                    old_size=(old_size, old_size),
                    num_prefix_tokens=0,
                )

            out_dict[k] = v
        else:
            out_dict[k] = v

    return out_dict

default_cfgs = generate_default_cfgs({
    'vit_base_patch16_224.orig_in21k': _cfg(
        # url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
        hf_hub_id='timm/',
        num_classes=0),
})

@register_model
def vit_base_patch16_224_cil(pretrained=False, **kwargs):
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)

    model = build_model_with_cfg(
        VisionTransformerDistilledCIL,
        'vit_base_patch16_224.orig_in21k',
        pretrained=False,
        **dict(model_args, **kwargs),
    )

    if pretrained:
        checkpoint_path = torch.hub.load_state_dict_from_url(
            'https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
            map_location='cpu',
            progress=True,
        )

        state_dict = checkpoint_path.get('model', checkpoint_path)
        state_dict = checkpoint_filter_fn_cil(state_dict, model)

        for k in ['head.weight', 'head.bias', 'pre_logits.fc.weight', 'pre_logits.fc.bias']:
            state_dict.pop(k, None)

        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

    return model