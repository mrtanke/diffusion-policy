# diffusion_policy/models/nn_utils.py
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """
    A technique from "Attention is All You Need" (Vaswani et al., 2017)
    
    Standard sinusoidal embedding for diffusion timestep t.
    Input: t (B,) int/float
    Output: (B, dim)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor: # t: (B,)
        device = t.device
        half = self.dim // 2
        emb_scale = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=device) * -emb_scale) # (half,)

        # t: (B,) -> (B, half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0) # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1) # (B, dim)
        if self.dim % 2 == 1: # zero pad
            emb = F.pad(emb, (0, 1), value=0.0)
        return emb # (B, dim)


class MLP(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
