from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.ops import FeaturePyramidNetwork


class ResNetFPNBackbone(nn.Module):
    def __init__(self, out_channels: int = 256, pretrained: bool = True):
        super().__init__()
        backbone = resnet50(weights="DEFAULT" if pretrained else None)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # C2
        self.layer2 = backbone.layer2  # C3
        self.layer3 = backbone.layer3  # C4
        self.layer4 = backbone.layer4  # C5

        in_channels_list = [256, 512, 1024, 2048]
        self.out_channels = out_channels
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            #extra_blocks=LastLevelMaxPool()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (B,3,H,W)
        return: OrderedDict of FPN features
        """
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        feats = OrderedDict()
        feats["0"] = c2
        feats["1"] = c3
        feats["2"] = c4
        feats["3"] = c5

        fpn_feats = self.fpn(feats)
        return fpn_feats
