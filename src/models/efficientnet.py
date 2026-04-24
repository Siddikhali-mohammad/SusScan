import timm
import torch.nn as nn


class TimmBinaryWrapper(nn.Module):
    # timm outputs [B, 1]; squeeze to [B] for BCEWithLogitsLoss
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).squeeze(-1)


class EFB4Binary(nn.Module):
    def __init__(self, pretrained=True, drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnet_b4", pretrained=pretrained,
                                           num_classes=0, global_pool='', drop_rate=drop_rate)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Dropout(drop_rate),
            nn.Linear(1792, 512),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


def build_efb4(pretrained=True, drop_rate=0.3, drop_path_rate=0.2):
    m = EFB4Binary(pretrained, drop_rate)
    backbone = list(m.backbone.parameters())
    head = list(m.head.parameters())
    return m, backbone, head
