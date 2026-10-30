import torch
from torch import nn
from copy import deepcopy


class LLL_Net_VitHydra(nn.Module):
    """vit-hydra backbone (1 cls + N dist tokens) + 1 + N head lists for CIL.

    cls_features    -> heads          (per task)
    dist_features_i -> heads_dist[i]  (per task), i = 0..N-1
                       (regnety160, convnext_b, swin_b, resnet101, resnet50 for the 5-teacher model)

    Backbone weights (vit_hydra_final.pth) loaded from backbone_weights_path; the backbone stays trainable.
    """

    def __init__(self, model, backbone_weights_path=None):
        super().__init__()
        self.model = model
        self.out_size = model.num_features
        self.num_dist = model.num_dist_tokens

        if backbone_weights_path is not None:
            self.model.load_state_dict(torch.load(backbone_weights_path, map_location='cpu'))
            print(f'Loaded vit-hydra backbone weights from {backbone_weights_path}')
        else:
            print('vit-hydra backbone randomly initialized')

        self.heads      = nn.ModuleList()
        self.heads_dist = nn.ModuleList([nn.ModuleList() for _ in range(self.num_dist)])

        self.task_cls    = []
        self.task_offset = []

    def add_head(self, num_outputs):
        self.heads.append(nn.Linear(self.out_size, num_outputs))
        for heads in self.heads_dist:
            heads.append(nn.Linear(self.out_size, num_outputs))
        self.task_cls    = torch.tensor([h.out_features for h in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def forward(self, x, return_features=False):
        assert len(self.heads) > 0, "No heads added yet"

        tokens    = self.model.forward_tokens(x)    # [B, 1 + N, D]
        cls_feat  = tokens[:, 0]
        dist_feat = [tokens[:, i + 1] for i in range(self.num_dist)]

        out = {
            "cls_logits":  [h(cls_feat) for h in self.heads],
            "dist_logits": [[h(f) for h in heads] for f, heads in zip(dist_feat, self.heads_dist)],
        }
        if return_features:
            out.update({
                "cls_features":  cls_feat,
                "dist_features": dist_feat,
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
