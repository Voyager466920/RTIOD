import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class Expert:
    def __init__(self, embed_dim:int=128, mlp_dim:int=128, dropout:float=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.relu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class MoEBlock:
    def __init__(self, embed_dim:int=128, mlp_dim:int=128, num_experts:int=6, experts_per_token:int=2, dropout:float=0.1):
        super().__init__()
        self.norm = RMSNorm(embed_dim)
        self.experts = nn.ModuleList(
            [Expert(embed_dim, mlp_dim, dropout) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(embed_dim, mlp_dim)
        self.gate_dropout = nn.Dropout(dropout)
        self.experts_per_token = experts_per_token
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.norm(x)
        gate_logits = self.gate(x_norm)
        gate_probs = F.softmax(gate_logits, dim=-1)
        gate_probs = self.gate_dropout(gate_probs)
        gate_probs = gate_probs / (gate_probs.sum(dim=-1, keepdim=True) + 1e-8)
        topk_probs, topk_idx = torch.topk(
            gate_probs, self.experts_per_token, dim=-1
        )

        B, S, D = x_norm.shape
        K = self.experts_per_token
        x_flat = x_norm.view(-1, D)
        topk_idx_flat = topk_idx.view(-1, K)
        topk_probs_flat = topk_probs.view(-1, K)
        out_flat = torch.zeros_like(x_flat, dtype=x_flat.dtype)

        for eid, expert in enumerate(self.experts):
            mask = (topk_idx_flat == eid)
            if not mask.any():
                continue
            rows = mask.any(dim=1).nonzero(as_tuple=False).squeeze(-1)
            expert_in = x_flat.index_select(0, rows)
            expert_out = expert(expert_in)
            probs = (topk_probs_flat[rows] * mask[rows].float()).sum(dim=1)
            expert_out.mul_(probs.unsqueeze(-1))
            out_flat.index_add_(0, rows, expert_out)

        output = out_flat.view(B, S, D)
        importance = gate_probs.float().sum(dim=(0, 1))
        load = torch.zeros(self.num_experts, device=x.device, dtype=torch.float32)
        load.scatter_add_(0, topk_idx_flat.reshape(-1),
                          torch.ones_like(topk_idx_flat.reshape(-1), dtype=load.dtype))
        balance_loss = 0.5 * (
                torch.std(importance / (importance.sum() + 1e-8)) +
                torch.std(load / (load.sum() + 1e-8))
        )

        return output, balance_loss



class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)