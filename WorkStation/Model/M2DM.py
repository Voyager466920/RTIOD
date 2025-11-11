import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicProjection(nn.Module):
    def __init__(self, feat_ch, aux_dim, rank=64, hidden=256, out_spatial=8):
        super().__init__()
        r = min(rank, max(4, feat_ch // 4))
        if r % 2 == 1: r += 1
        d = feat_ch + aux_dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.pre = nn.Sequential(nn.Linear(d, hidden), nn.ReLU())
        self.fc_base_c = nn.Linear(hidden, feat_ch)
        self.fc_base_s = nn.Linear(hidden, out_spatial)
        self.fc_phi_c = nn.Linear(hidden, r)
        self.fc_phi_s = nn.Linear(hidden, r)
        self.fc_Wc = nn.Linear(hidden, feat_ch * r)
        self.fc_Ws = nn.Linear(hidden, out_spatial * r)
        self.feat_ch = feat_ch
        self.out_spatial = out_spatial
        self.rank = r

    def fourier(self, u):
        b, r = u.shape
        half = r // 2
        w = torch.linspace(1., math.pi, steps=half, device=u.device, dtype=u.dtype).view(1, half)
        u = u[:, :half]
        s = torch.sin(u * w)
        c = torch.cos(u * w)
        return torch.cat([s, c], dim=-1)

    def forward(self, x, a):
        b,c,_,_ = x.shape
        g = self.gap(x).view(b,c)
        h = self.pre(torch.cat([g, a], dim=-1))
        m_base_c = self.fc_base_c(h)
        m_base_s = self.fc_base_s(h)
        phi_c = self.fourier(self.fc_phi_c(h))
        phi_s = self.fourier(self.fc_phi_s(h))
        Wc = self.fc_Wc(h).view(b, self.feat_ch, self.rank)
        Ws = self.fc_Ws(h).view(b, self.out_spatial, self.rank)
        m_proj_c = torch.bmm(Wc, phi_c.unsqueeze(-1)).squeeze(-1)
        m_proj_s = torch.bmm(Ws, phi_s.unsqueeze(-1)).squeeze(-1)
        c_m = m_base_c * m_proj_c
        s_m = m_base_s * m_proj_s
        return c_m, s_m

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

class M2DM(nn.Module):
    def __init__(self, feat_ch, aux_dim, rank=64, hidden=256):
        super().__init__()
        self.dp = DynamicProjection(feat_ch, aux_dim, rank=rank, hidden=hidden, out_spatial=8)
        self.sm = SpatialModulation()
    def forward(self, x, a):
        b,c,h,w = x.shape
        c_m, s_m = self.dp(x, a)
        ch_w = torch.sigmoid(c_m).view(b, c, 1, 1)
        xc = x * ch_w
        xs = self.sm(x, s_m)
        return x + xc + xs
