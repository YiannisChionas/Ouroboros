import torch
from torch import nn
from copy import deepcopy


class LLL_Net_Cls_Only(nn.Module):
    """
    Backbone-agnostic distillation network that feeds cls features to both heads.

    Ablation counterpart of LLL_Net_Distilled: instead of routing cls_features → heads
    and dist_features → heads_dist, both heads receive the same cls features.

    Handles any backbone output format:
    - single tensor  (plain ViT with num_classes=0 in timm)
    - tuple          (DeiT in timm returns (cls, dist))
    - dict           (custom backbones returning {"cls_features": ..., ...})
    """

    def __init__(self, model, remove_existing_head=False):

        head_var = model.head_var
        assert type(head_var) == str

        assert not remove_existing_head or hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear, nn.Identity], \
            "Given model's head {} is not an instance of nn.Sequential or nn.Linear".format(head_var)

        super(LLL_Net_Cls_Only, self).__init__()

        self.model = model
        self.out_size = model.num_features

        self.heads = nn.ModuleList()
        self.heads_dist = nn.ModuleList()
        self.task_cls = []
        self.task_offset = []
        self._initialize_weights()

    def _extract_cls_features(self, outputs_backbone):
        if isinstance(outputs_backbone, dict):
            return outputs_backbone["cls_features"]
        elif isinstance(outputs_backbone, (tuple, list)):
            return outputs_backbone[0]
        else:
            return outputs_backbone

    def add_head(self, num_outputs):
        self.heads.append(nn.Linear(self.out_size, num_outputs))
        self.heads_dist.append(nn.Linear(self.out_size, num_outputs))
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def forward(self, x, return_features=False):
        outputs_backbone = self.model(x)
        assert len(self.heads) > 0, "Cannot access any head"

        cls_features = self._extract_cls_features(outputs_backbone)

        cls_logits = [head(cls_features) for head in self.heads]
        dist_logits = [head(cls_features) for head in self.heads_dist]

        if return_features:
            return {
                "cls_logits": cls_logits,
                "dist_logits": dist_logits,
                "cls_features": cls_features,
            }
        return {
            "cls_logits": cls_logits,
            "dist_logits": dist_logits,
        }

    def get_copy(self):
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        self.load_state_dict(deepcopy(state_dict))

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        pass
