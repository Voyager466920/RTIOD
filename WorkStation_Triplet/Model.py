import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


from WorkStation_MoE.MMMMoE.Backbone import ImageExtractorResnet50

class SupConResNet50(nn.Module):
    def __init__(self, feat_dim=128, pretrained=True):
        super().__init__()
        self.backbone = ImageExtractorResnet50(pretrained=pretrained)  # c1,c2,c3,c4 뽑는 구조
        self.head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, feat_dim)
        )

    def forward(self, x):
        c1, c2, c3, c4 = self.backbone(x)
        h = F.adaptive_avg_pool2d(c4, 1).flatten(1)
        z = self.head(h)
        z = F.normalize(z, dim=1)
        return z
