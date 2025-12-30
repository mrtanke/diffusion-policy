# diffusion_policy/models/denoisers.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.models.nn_utils import SinusoidalPosEmb, MLP


class ResBlock1D(nn.Module):
    """
    FiLM from "FiLM: Visual Reasoning with a General Conditioning Layer" (Perez et al., 2018)"

    Residual 1D conv block with FiLM-style conditioning from (time + cond).
    It lets extra information control how feature maps are used inside the network.
    Input:
        x:    (B, dim, H)
        cond: (B, cond_dim)
    Output:
        (B, dim, H)
    """
    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, dim)
        self.conv1 = nn.Conv1d(dim, dim, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, dim)
        self.conv2 = nn.Conv1d(dim, dim, 3, padding=1)
        self.dropout = nn.Dropout(dropout)

        # produce scale and shift for FiLM: (B, 2*dim)
        self.film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * dim), # 2 * dim -> scale γ, shift β
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # h -> main feature stream
        h = self.conv1(F.silu(self.norm1(x)))

        scale_shift = self.film(cond).unsqueeze(-1)  # (B,2*dim,1)
        scale, shift = scale_shift.chunk(2, dim=1)   # each (B,dim,1)

        # FiLM(h)=h⋅(1+γ)+β
        h = h * (1.0 + scale) + shift
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return x + h


class TemporalUNetDenoiser(nn.Module):
    """
    UNet from "U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger et al., 2015)"

    Denoiser for action trajectories.
    Input:
      x_noisy: (B, H, action_dim)
      t:       (B,)
      cond:    (B, cond_dim)
    Output:
      eps_pred:(B, H, action_dim)
    """
    def __init__(
        self,
        action_dim: int = 2, # action dimension
        base_dim: int = 128, # base channel dimension
        cond_dim: int = 256, # condition dimension
        time_dim: int = 128, # time embedding dimension
        depth: int = 4, # down/up-sampling depth
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim = action_dim

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, time_dim)

        # input projection (action_dim -> base_dim)
        self.in_proj = nn.Conv1d(action_dim, base_dim, 1)

        # down path
        # downs -> conditioned ResBlocks
        # downsamples -> strided convs for downsampling
        dims = [base_dim] * depth
        self.downs = nn.ModuleList([ResBlock1D(d, cond_dim=time_dim, dropout=dropout) for d in dims])
        self.downsamples = nn.ModuleList([nn.Conv1d(base_dim, base_dim, 4, stride=2, padding=1) for _ in range(depth - 1)])

        # middle
        self.mid1 = ResBlock1D(base_dim, cond_dim=time_dim, dropout=dropout)
        self.mid2 = ResBlock1D(base_dim, cond_dim=time_dim, dropout=dropout)

        # up path
        self.upsamples = nn.ModuleList([nn.ConvTranspose1d(base_dim, base_dim, 4, stride=2, padding=1) for _ in range(depth - 1)])
        self.ups = nn.ModuleList([ResBlock1D(base_dim, cond_dim=time_dim, dropout=dropout) for _ in range(depth)])

        # output projection
        self.out_proj = nn.Sequential(
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.Conv1d(base_dim, action_dim, 1),
        )

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x_noisy: (B,H,action_dim) -> (B,action_dim,H)
        x = x_noisy.transpose(1, 2)

        # combine time + condition into a single conditioning vector
        t_emb = self.time_emb(t)                 # (B, time_dim)
        c_emb = self.cond_proj(cond)             # (B, time_dim)
        tc = t_emb + c_emb                       # (B, time_dim)

        x = self.in_proj(x)                      # (B, base_dim, H)

        skips = []
        for i, block in enumerate(self.downs):
            x = block(x, tc)
            skips.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        x = self.mid1(x, tc)
        x = self.mid2(x, tc)

        for i in range(len(self.ups)):
            if i < len(self.upsamples):
                x = self.upsamples[i](x)
            skip = skips.pop()

            # if length mismatch due to odd sizes, crop to match
            if x.shape[-1] != skip.shape[-1]:
                min_len = min(x.shape[-1], skip.shape[-1])
                x = x[..., :min_len]
                skip = skip[..., :min_len]
            
            # add skip connection
            x = x + skip
            x = self.ups[i](x, tc)

        eps = self.out_proj(x)                   # (B, action_dim, H)
        return eps.transpose(1, 2)               # (B, H, action_dim)