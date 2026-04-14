from torch import nn
import timm


class ResNet50_a1_in1k_224(nn.Module):
    """
    ResNet-50 pretrained (timm) as feature extractor, with a simple Linear head (fc)
    that FACIL/LLL_Net can remove and replace with incremental heads.
    """

    def __init__(self, num_classes=100, pretrained=False, backbone_name="resnet50.a1_in1k", freeze_backbone=False):
        super().__init__()

        # Create backbone. For timm ResNet, setting num_classes=0 makes it return features.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,   # feature extractor
        )

        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Get feature dimension safely (avoid hardcoding 2048)
        feat_dim = getattr(self.backbone, "num_features", None)

        if feat_dim is None:
            # Fallback: timm ResNet typically exposes a classifier named 'fc'
            # but since num_classes=0, it may be Identity. Safer to try num_features first.
            raise RuntimeError("Could not infer feature dim from timm model (missing .num_features).")

        # Standalone head (FACIL will remove/ignore it when remove_existing_head=True)
        self.fc = nn.Linear(feat_dim, num_classes, bias=True)

        # FACIL hook: tells LLL_Net which attribute is the head to remove
        self.head_var = "fc"

    def forward(self, x):
        feats = self.backbone(x)      # [B, feat_dim]
        logits = self.fc(feats)       # [B, num_classes]
        return logits

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()


def resnet50_imagenet_224(num_out=100, pretrained=False, variant="a1", freeze=False):
    """
    variant:
      - "a1"  -> resnet50.a1_in1k  (strong modern recipe)
      - "ra"  -> resnet50.ra_in1k
      - "ram" -> resnet50.ram_in1k
    """
    name_map = {
        "a1": "resnet50.a1_in1k",
        "ra": "resnet50.ra_in1k",
        "ram": "resnet50.ram_in1k",
    }
    if variant not in name_map:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(name_map.keys())}.")

    if pretrained:
        return ResNet50_a1_in1k_224(num_classes=num_out, pretrained=True, backbone_name=name_map[variant], freeze_backbone=freeze)
    else:
        raise NotImplementedError("This wrapper is intended for pretrained=True only.")
