import torch
import torch.nn as nn
import math

from WorkStation.AuxDetScratch.M2DM import M2DM


class AuxDetScratch(nn.Module):
    def __init__(self, meta_in_dim=9, meta_hidden=64, meta_out_dim=128):
        super().__init__()
        self.image_extractor = ImageExtractor()
        self.meta_encoder = MetadataEncoder(in_dim=meta_in_dim, hidden=meta_hidden, out_dim=meta_out_dim)
        self.cdown = CDown(in_channel=64, out_channel=512)
        self.fuse = AuxFusion(ch_delta=512, dim_meta=meta_out_dim, hidden=128, out_dim=128)
        self.m2dm = M2DM(meta_out_dim, meta_in_dim, hidden=256)
    def forward(self, img, meta):
        x1, x2, x3, x4 = self.image_extractor(img)
        z = self.meta_encoder(meta)
        #TODO meta data encoding 정리
        delx = self.cdown(x1, x4) - x4
        a = self.fuse(delx, z)
        x_bar = self.m2dm(a)
        return x_bar



class ImageExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
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
    def __init__(self, in_dim=9, hidden=64, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.ReLU()
        )
    def forward(self, meta):
        wind_dir = meta[:, 4] * math.pi / 180.0
        sin_dir = torch.sin(wind_dir).unsqueeze(1)
        cos_dir = torch.cos(wind_dir).unsqueeze(1)
        meta_proc = torch.cat([meta[:, :4], sin_dir, cos_dir, meta[:, 5:]], dim=1)
        return self.mlp(meta_proc)

class CDown(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU()

    def forward(self, x, ref):
        y = self.act(self.bn(self.conv(x)))
        #레퍼런스랑 다르면 interpolation 구현 전
        if y.shape[-2:] != ref.shape[-2:] : print(r"인풋 차원이 다릅니다")
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