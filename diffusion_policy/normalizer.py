# diffusion-policy/normalizer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from typing import Any


@dataclass
class MinMaxStats:
    min: np.ndarray
    max: np.ndarray


class MinMaxNormalizer:
    """
    Normalize vectors to [-1, 1] per-dimension using dataset min/max.
    """
    def __init__(self, stats: Dict[str, MinMaxStats]):
        self.stats = stats

    @staticmethod
    def compute_stats(x: np.ndarray) -> MinMaxStats:
        x2 = x.reshape(-1, x.shape[-1])
        return MinMaxStats(min=x2.min(axis=0), max=x2.max(axis=0))

    def normalize_np(self, key: str, x: np.ndarray) -> np.ndarray: # convert to [-1, 1]
        s = self.stats[key]
        denom = (s.max - s.min)
        denom = np.where(denom == 0, 1.0, denom) # safety check to avoid division by zero
        y = (x - s.min) / denom # scale to [0, 1]
        return y * 2.0 - 1.0 # scale to [-1, 1]

    def unnormalize_np(self, key: str, y: np.ndarray) -> np.ndarray:
        s = self.stats[key]
        x01 = (y + 1.0) / 2.0 # scale to [0, 1]
        return x01 * (s.max - s.min) + s.min # scale to original

    def normalize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        s = self.stats[key]
        smin = torch.as_tensor(s.min, device=x.device, dtype=x.dtype)
        smax = torch.as_tensor(s.max, device=x.device, dtype=x.dtype)
        denom = torch.where((smax - smin) == 0, torch.ones_like(smax), (smax - smin))
        y = (x - smin) / denom
        return y * 2.0 - 1.0

    def unnormalize(self, key: str, y: torch.Tensor) -> torch.Tensor:
        s = self.stats[key]
        smin = torch.as_tensor(s.min, device=y.device, dtype=y.dtype)
        smax = torch.as_tensor(s.max, device=y.device, dtype=y.dtype)
        x01 = (y + 1.0) / 2.0
        return x01 * (smax - smin) + smin

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for k, v in self.stats.items():
            out[k] = {"min": v.min, "max": v.max}
        return out
    
    @staticmethod
    def load_state_dict(state: Dict[str, Dict[str, Any]]) -> "MinMaxNormalizer":
        stats = {}
        for k, v in state.items():
            stats[k] = MinMaxStats(min=np.asarray(v["min"]), max=np.asarray(v["max"]))
        return MinMaxNormalizer(stats=stats)