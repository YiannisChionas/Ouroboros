"""ViT-Orth — ViT-S/16 with two prefix tokens (cls + dist), both trained on ground truth.

Same architecture as vit-hydra (`VisionTransformerHydra`) with N = 1 distillation token:

    token 0 (cls)  -> head          <- ground truth (CE)
    token 1 (dist) -> heads_dist[0] <- ground truth (CE)

Pretrained from scratch on the 200 mammal classes of iNat-500 (= iNatM-200) by
pretrain_vit_orth.py, with a cross-correlation penalty between the two tokens' features
(w_orth = 0 for the ablation). There is no teacher.
"""
from .vit_hydra_tokens import VisionTransformerHydra

__all__ = ['vit_orth_small_patch16_224', 'vit_orth_small_patch16_224_cil']


def vit_orth_small_patch16_224(num_classes=200, drop_path_rate=0.1, **kwargs):
    """ViT-S/16 @224 (DeiT-S config) with 1 cls + 1 dist token, random init."""
    return VisionTransformerHydra(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6,
        num_classes=num_classes, drop_path_rate=drop_path_rate,
        num_dist_tokens=1, **kwargs)


def vit_orth_small_patch16_224_cil(pretrained=False, **kwargs):
    """CIL backbone: ViT-Orth S/16 without drop path.

    The pretrained weights are loaded by LLL_Net_VitHydra (approach_args.backbone_weights).
    """
    assert not pretrained, 'vit-orth weights are loaded via approach_args.backbone_weights'
    return vit_orth_small_patch16_224(drop_path_rate=0.0, **kwargs)
