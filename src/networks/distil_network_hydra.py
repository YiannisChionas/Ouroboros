import torch
from torch import nn
from copy import deepcopy

from .mlp import MLP


class LLL_Net_Hydra(nn.Module):
    """DeiT backbone + 4 head lists for vit_hydra CIL.

    cls_features  -> heads          (per task)  — original
    dist_features -> heads_dist     (per task)  — original
    cls_features  -> mlp_cls -> heads_mlp_cls   (per task)  — pretrained from ConvNeXt
    dist_features -> mlp_dist -> heads_mlp_dist (per task)  — pretrained from Swin

    mlp_cls and mlp_dist are loaded from a pretrained checkpoint and kept frozen.
    """

    def __init__(self, model, mlp_weights_path):
        super().__init__()
        self.model = model
        self.out_size = model.num_features

        self.mlp_cls  = MLP(self.out_size)
        self.mlp_dist = MLP(self.out_size)

        ckpt = torch.load(mlp_weights_path, map_location='cpu')
        self.mlp_cls.load_state_dict(ckpt['mlp_cls'])
        self.mlp_dist.load_state_dict(ckpt['mlp_dist'])
        for p in self.mlp_cls.parameters():  p.requires_grad = False
        for p in self.mlp_dist.parameters(): p.requires_grad = False

        self.heads          = nn.ModuleList()
        self.heads_dist     = nn.ModuleList()
        self.heads_mlp_cls  = nn.ModuleList()
        self.heads_mlp_dist = nn.ModuleList()

        self.task_cls    = []
        self.task_offset = []

    def add_head(self, num_outputs):
        self.heads.append(nn.Linear(self.out_size, num_outputs))
        self.heads_dist.append(nn.Linear(self.out_size, num_outputs))
        self.heads_mlp_cls.append(nn.Linear(self.out_size, num_outputs))
        self.heads_mlp_dist.append(nn.Linear(self.out_size, num_outputs))
        self.task_cls    = torch.tensor([h.out_features for h in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def forward(self, x, return_features=False):
        assert len(self.heads) > 0, "No heads added yet"

        cls_feat, dist_feat = self.model(x)
        mlp_cls_feat  = self.mlp_cls(cls_feat)
        mlp_dist_feat = self.mlp_dist(dist_feat)

        cls_logits      = [h(cls_feat)       for h in self.heads]
        dist_logits     = [h(dist_feat)      for h in self.heads_dist]
        mlp_cls_logits  = [h(mlp_cls_feat)  for h in self.heads_mlp_cls]
        mlp_dist_logits = [h(mlp_dist_feat) for h in self.heads_mlp_dist]

        out = {
            "cls_logits":      cls_logits,
            "dist_logits":     dist_logits,
            "mlp_cls_logits":  mlp_cls_logits,
            "mlp_dist_logits": mlp_dist_logits,
        }
        if return_features:
            out.update({
                "cls_features":      cls_feat,
                "dist_features":     dist_feat,
                "mlp_cls_features":  mlp_cls_feat,
                "mlp_dist_features": mlp_dist_feat,
            })
        return out

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
