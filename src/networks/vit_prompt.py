"""ViT backbone with learnable prompt pool (L2P).

ViT_Prompt subclasses timm's VisionTransformer and restructures forward_features
to match the PILOT / original L2P design:

  patch_embed → [select prompts] → cat [prompts, patches]
             → prepend CLS → add extended pos_embed → transformer blocks

Key design points vs a naive implementation:
  - pos_embed is extended to cover prompt token positions (1+K*L+N instead of 1+N).
    Prompt positions are initialised with the CLS positional embedding, following
    PILOT's resize_pos_embed logic.  The pretrained filter handles this resize so
    that CLS and patch positions are loaded correctly from the pretrained checkpoint.
  - A frozen reference backbone extracts the query for prompt selection (mirrors
    PILOT's original_backbone / PromptVitNet).  Without it, query = mean patch
    embedding (content-aware fallback, but weaker).
  - self.last_reduce_sim is written after every forward so the approach can add
    the pull-constraint to the loss.
"""
import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer, checkpoint_filter_fn
from timm.models._builder import build_model_with_cfg
from timm.models._registry import register_model
from timm.models._manipulate import checkpoint_seq

from .prompt import PromptPool

__all__ = ['ViT_Prompt']


# ------------------------------------------------------------------ checkpoint filter

def _l2p_filter(state_dict, model):
    """Standard ViT filter + extend pos_embed for K*L prompt positions.

    Prompt slots are filled with the CLS positional embedding (PILOT convention).
    Pretrained CLS and patch positions are preserved exactly.
    """
    state_dict = checkpoint_filter_fn(state_dict, model)

    if 'pos_embed' in state_dict:
        pe_ckpt  = state_dict['pos_embed']   # (1, 1+N, D) from pretrained
        pe_model = model.pos_embed           # (1, 1+K*L+N, D) — already extended
        if pe_ckpt.shape != pe_model.shape:
            n_prompt  = pe_model.shape[1] - pe_ckpt.shape[1]
            cls_pos   = pe_ckpt[:, :1, :]
            patch_pos = pe_ckpt[:, 1:, :]
            # Repeat CLS position for all prompt slots (PILOT convention)
            prompt_pos = cls_pos.expand(1, n_prompt, -1)
            state_dict['pos_embed'] = torch.cat([cls_pos, prompt_pos, patch_pos], dim=1)

    return state_dict


# ------------------------------------------------------------------ model

class ViT_Prompt(VisionTransformer):
    """ViT with learnable prompt pool, following the PILOT / L2P forward order.

    Args:
        pool_size    (int): prompts in the pool (M)
        top_k        (int): prompts selected per forward (K)
        prompt_len   (int): tokens per prompt (L)
        ref_backbone (str): timm model name for the frozen query backbone.
                            Defaults to None (falls back to mean patch embedding).
    """

    def __init__(self, pool_size=30, top_k=5, prompt_len=1, ref_backbone=None, **kwargs):
        super().__init__(**kwargs)
        self.prompt_pool      = PromptPool(pool_size, top_k, prompt_len, self.embed_dim)
        self.last_reduce_sim  = None
        self._pending_query   = None
        self._num_prompt_tokens = top_k * prompt_len

        # Extend pos_embed: (1, 1+N, D) → (1, 1+K*L+N, D)
        # Prompt slots initialised with CLS position; pretrained_filter_fn will
        # overwrite with correct values when loading a pretrained checkpoint.
        with torch.no_grad():
            cls_pos    = self.pos_embed[:, :1, :]
            patch_pos  = self.pos_embed[:, 1:, :]
            prompt_pos = cls_pos.expand(1, self._num_prompt_tokens, -1).clone()
            self.pos_embed = nn.Parameter(
                torch.cat([cls_pos, prompt_pos, patch_pos], dim=1)
            )

        # Frozen reference backbone for query extraction (PILOT's original_backbone)
        self._ref_backbone = None
        if ref_backbone is not None:
            self._ref_backbone = timm.create_model(ref_backbone, pretrained=True, num_classes=0)
            for p in self._ref_backbone.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------ mode

    def train(self, mode: bool = True):
        """Keep the frozen reference backbone permanently in eval mode."""
        super().train(mode)
        if self._ref_backbone is not None:
            self._ref_backbone.eval()
        return self

    # ------------------------------------------------------------------ forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run frozen ref backbone to get query, then normal ViT forward."""
        if self._ref_backbone is not None:
            with torch.no_grad():
                self._pending_query = self._ref_backbone(x)   # (B, D)
        return super().forward(x)

    def forward_features(self, x, attn_mask=None):
        x = self.patch_embed(x)    # (B, N, D) — patches only, no CLS yet
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        # Query: from frozen ref backbone (preferred) or mean of patch embeddings
        query = self._pending_query if self._pending_query is not None else x.mean(dim=1)
        self._pending_query = None

        prompts, reduce_sim = self.prompt_pool(query)   # (B, K*L, D), scalar
        self.last_reduce_sim = reduce_sim

        # PILOT order: [prompts, patches] → prepend CLS → add extended pos_embed
        x = torch.cat([prompts, x], dim=1)              # (B, K*L+N, D)
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)           # (B, 1+K*L+N, D)

        if attn_mask is not None:
            for blk in self.blocks:
                x = blk(x, attn_mask=attn_mask)
        elif self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        return x

    # forward_head unchanged — global_pool='token' → x[:,0] (CLS) → ignored prompts ✓


# ------------------------------------------------------------------ model registry

@register_model
def vit_small_patch16_224_prompt(pretrained=False, **kwargs):
    """ViT-Small/16 (augreg_in1k) with prompt pool.

    Reference backbone: vit_small_patch16_224.augreg_in1k (frozen, for query).
    """
    prompt_keys = ('pool_size', 'top_k', 'prompt_len', 'ref_backbone')
    prompt_kwargs = {k: kwargs.pop(k) for k in prompt_keys if k in kwargs}
    prompt_kwargs.setdefault('ref_backbone', 'vit_small_patch16_224.augreg_in1k')

    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, num_classes=0)
    return build_model_with_cfg(
        ViT_Prompt,
        'vit_small_patch16_224.augreg_in1k',
        pretrained=pretrained,
        pretrained_strict=False,
        pretrained_filter_fn=_l2p_filter,
        **dict(model_args, **prompt_kwargs, **kwargs),
    )


@register_model
def vit_base_patch16_224_prompt(pretrained=False, **kwargs):
    """ViT-Base/16 (augreg_in21k) with prompt pool.

    Reference backbone: vit_base_patch16_224.augreg_in21k (frozen, for query).
    """
    prompt_keys = ('pool_size', 'top_k', 'prompt_len', 'ref_backbone')
    prompt_kwargs = {k: kwargs.pop(k) for k in prompt_keys if k in kwargs}
    prompt_kwargs.setdefault('ref_backbone', 'vit_base_patch16_224.augreg_in21k')

    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=0)
    return build_model_with_cfg(
        ViT_Prompt,
        'vit_base_patch16_224.augreg_in21k',
        pretrained=pretrained,
        pretrained_strict=False,
        pretrained_filter_fn=_l2p_filter,
        **dict(model_args, **prompt_kwargs, **kwargs),
    )
