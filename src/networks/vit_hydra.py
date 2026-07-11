import torch
import torch.nn as nn

from .mlp import MLP


class VitHydra(nn.Module):
    """DeiT backbone (frozen) + 2 MLP projectors for mini-pretraining.

    cls_features  -> mlp_cls -> fc_cls  (distilled from Teacher A)
    dist_features -> mlp_dist -> fc_dist (distilled from Teacher B)

    After pretraining, fc_cls/fc_dist are discarded — only mlp_cls/mlp_dist are kept.
    """

    def __init__(self, backbone, teacher_out_dim):
        super().__init__()
        self.backbone = backbone
        embed_dim = backbone.num_features

        self.mlp_cls  = MLP(embed_dim)
        self.mlp_dist = MLP(embed_dim)

        self.fc_cls  = nn.Linear(embed_dim, teacher_out_dim)
        self.fc_dist = nn.Linear(embed_dim, teacher_out_dim)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        cls_features, dist_features = self.backbone(x)

        mlp_cls_out  = self.mlp_cls(cls_features)
        mlp_dist_out = self.mlp_dist(dist_features)

        return {
            "cls_features":   cls_features,
            "dist_features":  dist_features,
            "fc_cls_logits":  self.fc_cls(mlp_cls_out),
            "fc_dist_logits": self.fc_dist(mlp_dist_out),
        }

    def save_mlps(self, path):
        torch.save({
            "mlp_cls":  self.mlp_cls.state_dict(),
            "mlp_dist": self.mlp_dist.state_dict(),
        }, path)
