# diffusion_policy/models/encoders.py
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """
    Lightweight image encoder for 96x96 PushT observations.
    A simple convnet followed by global average pooling and a linear layer.
    Input:  (B*To, 3, 96, 96)
    Output: (B*To, feat_dim)
    """
    def __init__(self, in_ch: int = 3, feat_dim: int = 256):
        super().__init__()
        # Simple strided conv stack -> global average pool -> linear
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2),  # 96 -> 48
            nn.GroupNorm(4, 32),
            nn.SiLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),     # 48 -> 24
            nn.GroupNorm(8, 64),
            nn.SiLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),    # 24 -> 12
            nn.GroupNorm(8, 128),
            nn.SiLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),   # 12 -> 6
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )
        self.proj = nn.Linear(256, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)                      # (B,256,6,6)
        h = h.mean(dim=(-2, -1))              # GAP -> (B,256)
        return self.proj(h)                   # (B,feat_dim)


class ObsConditionEncoder(nn.Module):
    """
    Efficient conditioning:
      Encode observation history once -> condition vector.
    Inputs:
      obs_image:     (B, To, 3, H, W) in [0,1]
      obs_agent_pos: (B, To, 2) in [-1,1]
    Output:
      cond:          (B, cond_dim)
    """
    def __init__(
        self,
        image_feat_dim: int = 256,
        lowdim_feat_dim: int = 64, # the dimension of fusion for agent pos
        cond_dim: int = 256,
        n_obs_steps: int = 2,
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps

        self.image_enc = SmallCNN(in_ch=3, feat_dim=image_feat_dim)
        self.lowdim_enc = nn.Sequential(
            nn.Linear(2 * n_obs_steps, 128),
            nn.SiLU(),
            nn.Linear(128, lowdim_feat_dim),
        )

        self.fuse = nn.Sequential(
            nn.Linear(image_feat_dim + lowdim_feat_dim, 512),
            nn.SiLU(),
            nn.Linear(512, cond_dim),
        )

    def forward(self, obs_image: torch.Tensor, obs_agent_pos: torch.Tensor) -> torch.Tensor:
        B, To, C, H, W = obs_image.shape
        assert To == self.n_obs_steps, f"Expected To={self.n_obs_steps}, got {To}"

        # image encode each obs frame, then average over time
        x = obs_image.reshape(B * To, C, H, W)
        img_feat = self.image_enc(x).reshape(B, To, -1).mean(dim=1)  # (B, image_feat_dim)

        # low-dim: flatten history
        low = obs_agent_pos.reshape(B, To * obs_agent_pos.shape[-1])  # (B, 2*To)
        low_feat = self.lowdim_enc(low)                               # (B, lowdim_feat_dim)

        cond = self.fuse(torch.cat([img_feat, low_feat], dim=-1))     # (B, cond_dim)
        return cond
