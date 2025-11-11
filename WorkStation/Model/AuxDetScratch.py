import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from WorkStation.Model.M2DM import M2DM


class AuxDetScratch(nn.Module):
    def __init__(self, meta_in_dim=10, meta_hidden=64, meta_out_dim=128, fpn_out=256):
        super().__init__()
        self.backbone = ImageExtractorResnet50()
        self.meta_encoder = MetadataEncoder(in_dim=meta_in_dim, hidden=meta_hidden, out_dim=meta_out_dim)
        self.cdown = CDown(in_channel=64, out_channel=512)
        self.fuse = AuxFusion(ch_delta=512, dim_meta=meta_out_dim, hidden=128, out_dim=128)
        self.m2dm1 = M2DM(feat_ch=64, aux_dim=128, rank=64, hidden=256)
        self.fpn = FeaturePyramidNetwork(in_channels_list=[64,128,256,512],
                                         out_channels=fpn_out,
                                         extra_blocks=LastLevelMaxPool())
    def forward(self, img, meta):
        xi, x1, x2, x3, x4 = self.backbone(img)
        z = self.meta_encoder(meta)
        delx = self.cdown(xi, x4) - x4
        a = self.fuse(delx, z)
        y1 = self.m2dm1(xi, a)
        feats = OrderedDict([("c1", y1), ("c2", x2), ("c3", x3), ("c4", x4)])
        p = self.fpn(feats)
        return p

class ImageExtractorResnet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        w = backbone.conv1.weight.data
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.conv1.weight.data = w.mean(dim=1, keepdim=True)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
    def forward(self, x):
        initial  = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(initial)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return  initial, x1, x2, x3, x4

# 논문에 나와있는 구조로 구현한 CNN
class ImageExtractorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

    def forward(self, x):
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return c1, c2, c3, c4


class MetadataEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=128):
        super().__init__()
        self.hidden = hidden
        self.out_dim = out_dim
        self.mlp = None

    def forward(self, meta):
        wind_dir = meta[:, 4] * math.pi / 180.0
        sin_dir = torch.sin(wind_dir).unsqueeze(1)
        cos_dir = torch.cos(wind_dir).unsqueeze(1)
        meta_proc = torch.cat([meta[:, :4], sin_dir, cos_dir, meta[:, 5:]], dim=1)
        if self.mlp is None:
            self.mlp = nn.Sequential(
                nn.Linear(meta_proc.size(1), self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, self.out_dim),
                nn.ReLU()
            ).to(meta_proc.device)
        return self.mlp(meta_proc)


class CDown(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, 2, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU()
        self.adap = nn.AdaptiveAvgPool2d

    def forward(self, x, ref):
        y = self.act(self.bn(self.conv(x)))
        if y.shape[-2:] != ref.shape[-2:]:
            y = self.adap(ref.shape[-2:])(y)
        return y


class ResidualFusion(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
    def forward(self, x):
        y = self.fc2(self.act(self.fc1(x)))
        return self.act(y + self.proj(x))

class AuxFusion(nn.Module):
    def __init__(self, ch_delta, dim_meta, hidden=128, out_dim=128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fuse = ResidualFusion(ch_delta + dim_meta, hidden, out_dim)
    def forward(self, delta_x, z):
        b,c,_,_ = delta_x.shape
        d = self.gap(delta_x).view(b,c)
        return self.fuse(torch.cat([d, z], dim=-1))