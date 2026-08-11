import torch
import torch.nn as nn

from .mlp import MLP


class VitHydra_v2(nn.Module):
    """DeiT backbone (frozen) + 4 MLP projectors for mini-pretraining.

    cls_features  -> mlp_cls -> fc_cls   (Teacher A: ConvNeXt-Base)
    dist_features -> mlp_dist -> fc_dist (Teacher B: Swin-Base)
    cls_features  -> mlp_res -> fc_res   (Teacher C: ResNet101)
    dist_features -> mlp_eff -> fc_eff   (Teacher D: EfficientNet-B4)

    After pretraining, fc_* heads are discarded — only mlp_* are kept.
    """

    def __init__(self, backbone, teacher_out_dim):
        super().__init__()
        self.backbone = backbone
        embed_dim = backbone.num_features

        self.mlp_cls  = MLP(embed_dim)
        self.mlp_dist = MLP(embed_dim)
        self.mlp_res  = MLP(embed_dim)
        self.mlp_eff  = MLP(embed_dim)

        self.fc_cls  = nn.Linear(embed_dim, teacher_out_dim)
        self.fc_dist = nn.Linear(embed_dim, teacher_out_dim)
        self.fc_res  = nn.Linear(embed_dim, teacher_out_dim)
        self.fc_eff  = nn.Linear(embed_dim, teacher_out_dim)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        cls_feat, dist_feat = self.backbone(x)

        mlp_cls_out  = self.mlp_cls(cls_feat)
        mlp_dist_out = self.mlp_dist(dist_feat)
        mlp_res_out  = self.mlp_res(cls_feat)
        mlp_eff_out  = self.mlp_eff(dist_feat)

        return {
            "cls_features":   cls_feat,
            "dist_features":  dist_feat,
            "fc_cls_logits":  self.fc_cls(mlp_cls_out),
            "fc_dist_logits": self.fc_dist(mlp_dist_out),
            "fc_res_logits":  self.fc_res(mlp_res_out),
            "fc_eff_logits":  self.fc_eff(mlp_eff_out),
        }

    def save_mlps(self, path):
        torch.save({
            "mlp_cls":  self.mlp_cls.state_dict(),
            "mlp_dist": self.mlp_dist.state_dict(),
            "mlp_res":  self.mlp_res.state_dict(),
            "mlp_eff":  self.mlp_eff.state_dict(),
        }, path)
