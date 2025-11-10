import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialModulation(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    def _parse(self, w):
        B = w.size(0)
        p = w[:, :8].view(B, 2, 4)
        k = torch.zeros(B, 2, 3, 3, device=w.device, dtype=w.dtype)
        k[:, :, 0, 1] = p[:, :, 0]
        k[:, :, 2, 1] = p[:, :, 1]
        k[:, :, 1, 0] = p[:, :, 2]
        k[:, :, 1, 2] = p[:, :, 3]
        k[:, :, 1, 1] = -(p.sum(dim=2))
        return k
    def forward(self, x, w):
        B, C, H, W = x.shape
        a = torch.mean(x, dim=1, keepdim=True)
        m, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([a, m], dim=1).view(1, B*2, H, W)
        k = self._parse(w).view(B, 2, 3, 3)
        y = F.conv2d(y, k, padding=1, groups=B)
        y = y.view(B, 1, H, W)
        return x * self.sigmoid(y)

class CondMLP(nn.Module):
    def __init__(self, feat_ch, aux_dim, hidden=256, out_extra=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(feat_ch + aux_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feat_ch + out_extra)
        )
    def forward(self, x, z):
        b, c, _, _ = x.shape
        g = self.gap(x).view(b, c)
        return self.mlp(torch.cat([g, z], dim=-1))

class M2DM(nn.Module):
    def __init__(self, feat_ch, aux_dim, hidden=256):
        super().__init__()
        self.cond = CondMLP(feat_ch, aux_dim, hidden, out_extra=8)
        self.sm = SpatialModulation()
    def forward(self, x, z):
        b, c, h, w = x.shape
        w_all = self.cond(x, z)
        ch_w = torch.sigmoid(w_all[:, :c]).view(b, c, 1, 1)
        sp_w = w_all[:, c:]
        xc = x * ch_w
        xs = self.sm(x, sp_w)
        return x + xc + xs
