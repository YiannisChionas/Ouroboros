import torch
import torch.nn as nn

from .mlp import MLP


class VitHydra_v3(nn.Module):
    """DeiT backbone (frozen) + 4 MLP projectors with varied architectures.

    cls_features  -> mlp_cls  (standard: 768→3072→768, 2 layers)  Teacher A: ConvNeXt-Base
    dist_features -> mlp_dist (narrow:   768→1536→768, 2 layers)  Teacher B: Swin-Base
    cls_features  -> mlp_res  (wide:     768→6144→768, 2 layers)  Teacher C: ResNet101
    dist_features -> mlp_eff  (deeper:   768→3072→3072→768, 3 layers) Teacher D: EfficientNet-B4

    After pretraining, fc_* heads are discarded — only mlp_* are kept.
    """

    def __init__(self, backbone, teacher_out_dim):
        super().__init__()
        self.backbone = backbone
        embed_dim = backbone.num_features

        self.mlp_cls  = MLP(embed_dim, hidden_dim=4 * embed_dim, num_layers=2)
        self.mlp_dist = MLP(embed_dim, hidden_dim=2 * embed_dim, num_layers=2)
        self.mlp_res  = MLP(embed_dim, hidden_dim=8 * embed_dim, num_layers=2)
        self.mlp_eff  = MLP(embed_dim, hidden_dim=4 * embed_dim, num_layers=3)

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
