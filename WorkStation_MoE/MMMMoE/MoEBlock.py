import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEBlock(nn.Module):
    def __init__(self, embed_dim:int=128, mlp_dim:int=128, num_experts:int=6, dropout:float=0.1):
        super().__init__()
        self.gate_dropout = nn.Dropout(dropout)
        self.num_experts = num_experts

        self.norm = RMSNorm(embed_dim)
        self.experts = nn.ModuleList(
            [Expert(embed_dim, mlp_dim) for _ in range(num_experts)]
        )
        self.gate = Gate(num_experts=num_experts, meta_dim=9, hidden_dim=128)

    def forward(self, moe_c4, meta):
        gate_probs = self.gate(meta)
        top1_idx = gate_probs.argmax(dim=-1)

        out = torch.zeros_like(moe_c4)
        for eid, expert in enumerate(self.experts):
            mask = (top1_idx == eid)
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=False).squeeze(-1)

            expert_in = moe_c4[idx]
            expert_out = expert(expert_in)

            out[idx] = expert_out

        importance = gate_probs.sum(dim=0) / (gate_probs.sum() + 1e-8)
        balance_loss = torch.std(importance)

        return out, balance_loss



class Gate(nn.Module):
    def __init__(self, num_experts: int, meta_dim:int=9, hidden_dim:int=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
    def forward(self, meta):
        logits = self.mlp(meta)
        probs = F.softmax(logits, dim=-1)
        return probs



class Expert(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)