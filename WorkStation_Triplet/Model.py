import torch.nn as nn
from torchvision import models
import torch.nn.functional as F



class SupConResNet50(nn.Module):
    def __init__(self, feat_dim=128, pretrained=True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(1, conv1.out_channels, kernel_size=conv1.kernel_size,
                                 stride=conv1.stride, padding=conv1.padding, bias=False)
        if pretrained:
            resnet.conv1.weight.data = conv1.weight.data.mean(dim=1, keepdim=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, feat_dim)
        )

    def forward(self, x):
        h = self.encoder(x).flatten(1)
        z = self.head(h)
        z = F.normalize(z, dim=1)
        return z