"""ViT backbone with learnable prompt pool (L2P).

ViT_Prompt subclasses timm's VisionTransformer and overrides forward_features
to inject prompt tokens before the transformer blocks.  After each forward,
self.last_reduce_sim holds the mean query-key cosine similarity, which the
approach adds to the loss as a pull constraint.
"""
import torch
from timm.models.vision_transformer import VisionTransformer
from timm.models._builder import build_model_with_cfg
from timm.models._registry import register_model
from timm.models._manipulate import checkpoint_seq

from .prompt import PromptPool

__all__ = ['ViT_Prompt']


class ViT_Prompt(VisionTransformer):
    """ViT with a prompt pool inserted before the transformer blocks."""

    def __init__(self, pool_size=30, top_k=5, prompt_len=1, **kwargs):
        super().__init__(**kwargs)
        self.prompt_pool    = PromptPool(pool_size, top_k, prompt_len, self.embed_dim)
        self.last_reduce_sim = None

    def forward_features(self, x, attn_mask=None):
        x = self.patch_embed(x)
        x = self._pos_embed(x)    # (B, 1+N, D)  — cls at position 0
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        # Query = cls token; select prompts from pool
        query = x[:, 0]                                    # (B, D)
        prompts, reduce_sim = self.prompt_pool(query)      # (B, K*L, D), scalar
        self.last_reduce_sim = reduce_sim

        # [cls, prompt_1..K*L, patch_1..N]
        x = torch.cat([x[:, :1], prompts, x[:, 1:]], dim=1)

        if attn_mask is not None:
            for blk in self.blocks:
                x = blk(x, attn_mask=attn_mask)
        elif self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        return x

    # forward_head unchanged — global_pool='token' extracts x[:,0] (cls), so
    # prompt tokens at positions 1..K*L are automatically ignored.


@register_model
def vit_small_patch16_224_prompt(pretrained=False, **kwargs):
    """ViT-Small (augreg_in1k) with learnable prompt pool."""
    prompt_keys = ('pool_size', 'top_k', 'prompt_len')
    prompt_kwargs = {k: kwargs.pop(k) for k in prompt_keys if k in kwargs}
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = build_model_with_cfg(
        ViT_Prompt,
        'vit_small_patch16_224.augreg_in1k',
        pretrained=pretrained,
        pretrained_strict=False,
        **dict(model_args, **prompt_kwargs, **kwargs),
    )
    return model
