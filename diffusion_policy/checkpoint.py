# diffusion_policy/checkpoint.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    ema_model: Optional[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    step: int,
    normalizer_state: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "normalizer": normalizer_state,
    }
    if ema_model is not None:
        ckpt["ema_model"] = ema_model.state_dict()
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    ema_model: Optional[torch.nn.Module],
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if ema_model is not None and "ema_model" in ckpt:
        ema_model.load_state_dict(ckpt["ema_model"], strict=True)
    return ckpt
