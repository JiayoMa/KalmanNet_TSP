import torch
import torch.nn as nn
import torchvision.models as models


class SpatiotemporalEncoder(nn.Module):

    def __init__(self, latent_dim=128, backbone="resnet18", pretrained=True):
        super().__init__()
        self.latent_dim = latent_dim

        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            feat_dim = 512
        elif backbone == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if pretrained:
            orig_conv1_weight = base.conv1.weight.clone()

        base.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                base.conv1.weight[:, :3, :, :] = orig_conv1_weight
                base.conv1.weight[:, 3:, :, :] = orig_conv1_weight

        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, img_prev, img_curr):
        x = torch.cat([img_prev, img_curr], dim=1)
        feat = self.backbone(x).flatten(1)
        z = self.projection(feat)
        return z
