from torchvision import models as tv_models
from timm import models as timm_models

from .deit_original import VisionTransformerDistilled, deit_small_distilled_patch16_224_cil, deit_base_distilled_patch16_224_cil
from .vit_dist import VisionTransformerDistilledCIL, vit_small_patch16_224_dist

# available torchvision models
tvmodels = [
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
            'wide_resnet50_2', 'wide_resnet101_2'
           ]

# available timm models
timmmodels = ['vit_base_patch16_224.orig_in21k', 'vit_base_patch16_224.augreg_in21k', 'vit_small_patch16_224.augreg_in1k', 'resnet50.a1_in1k','resnetv2_101x1_bit.goog_in21k']

# DeiT / ViT-dist Feature extractor overrides
allmodels = tvmodels + timmmodels + ['deit_small_distilled_patch16_224_cil', 'deit_base_distilled_patch16_224_cil', 'vit_small_patch16_224_dist']

def set_model_head_var(model):
    # ResNet
    if type(model) == tv_models.ResNet:
        model.head_var = 'fc'
    elif type(model) == timm_models.resnet.ResNet:
        model.head_var = 'fc'
    elif type(model) == timm_models.resnetv2.ResNetV2:
        model.head_var = "head.fc"
    # ViT (plain timm)
    elif type(model) == timm_models.vision_transformer.VisionTransformer:
        model.head_var = 'head'
    # DeiT pretrained (VisionTransformerDistilled) or ViT+noise dist_token (VisionTransformerDistilledCIL)
    elif type(model) in (VisionTransformerDistilled, VisionTransformerDistilledCIL):
        model.head_var = 'head'
    else:
        raise ModuleNotFoundError