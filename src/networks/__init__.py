from torchvision import models as tv_models
from timm import models as timm_models

from .lenet import LeNet
from .vggnet import VggNet
from .resnet32 import resnet32
from .deit_cil import vit_base_patch16_224_cil
from .deit_original import deit_small_distilled_patch16_224_cil, deit_base_distilled_patch16_224_cil
from .deit_cil import VisionTransformerDistilledCIL
from .deit_original import VisionTransformerDistilled

# available torchvision models
tvmodels = ['alexnet',
            'densenet121', 'densenet169', 'densenet201', 'densenet161',
            'googlenet',
            'inception_v3',
            'mobilenet_v2',
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
            'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
            'squeezenet1_0', 'squeezenet1_1',
            'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',
            'wide_resnet50_2', 'wide_resnet101_2'
            ]
# available timm models
timmmodels = ['vit_base_patch16_224.orig_in21k', 'vit_base_patch16_224.augreg_in21k', 'vit_small_patch16_224.augreg_in1k', 'resnet50.a1_in1k','resnetv2_101x1_bit.goog_in21k']

allmodels = tvmodels + timmmodels + ['resnet32', 'LeNet', 'VggNet','vit_base_patch16_224_cil', 'deit_small_distilled_patch16_224_cil', 'deit_base_distilled_patch16_224_cil']

def set_model_head_var(model):
    if type(model) == tv_models.AlexNet:
        model.head_var = 'classifier'
    elif type(model) == tv_models.DenseNet:
        model.head_var = 'classifier'
    elif type(model) == tv_models.Inception3:
        model.head_var = 'fc'
    elif type(model) == tv_models.ResNet:
        model.head_var = 'fc'
    elif type(model) == tv_models.VGG:
        model.head_var = 'classifier'
    elif type(model) == tv_models.GoogLeNet:
        model.head_var = 'fc'
    elif type(model) == tv_models.MobileNetV2:
        model.head_var = 'classifier'
    elif type(model) == tv_models.ShuffleNetV2:
        model.head_var = 'fc'
    elif type(model) == tv_models.SqueezeNet:
        model.head_var = 'classifier'
    elif type(model) == timm_models.vision_transformer.VisionTransformer:
        model.head_var = 'head'
    elif type(model) == timm_models.resnet.ResNet:
        model.head_var = 'fc'
    elif type(model) == timm_models.resnetv2.ResNetV2:
        model.head_var = "head.fc"
    elif type(model) == VisionTransformerDistilledCIL:
        model.head_var = "head"
    elif type(model) == VisionTransformerDistilled:
        model.head_var = "head"
    else:
        raise ModuleNotFoundError
