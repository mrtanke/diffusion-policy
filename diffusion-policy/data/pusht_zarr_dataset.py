# diffusion-policy/data/pusht_zarr_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr

from diffusion_policy.data.sequence_utils import create_sample_indices, sample_sequence
from diffusion_policy.normalizer import MinMaxNormalizer, MinMaxStats


@dataclass(frozen=True)
class PushTWindowSpec:
    n_obs_steps: int = 2 # number of observation steps (To) = current position + previous position
    horizon: int = 10 # total horizon (H) = n_obs_steps + n_action_steps
    n_action_steps: int = 8 # number of action steps (Ta)
    pad_before: int = 1   # = n_obs_steps - 1
    pad_after: int = 7    # = n_action_steps - 1


class PushTImageZarrDataset(Dataset):
    """
    Translate raw files on the hard drive into structured data samples.
    Loads PushT replay from a zarr group and returns:
      obs.image:      (To, 3, H, W) in [0,1]
      obs.agent_pos:  (To, 2) normalized to [-1,1]
      action:         (H, 2) normalized to [-1,1]
    """
    def __init__(
        self,
        zarr_path: str,
        window: PushTWindowSpec = PushTWindowSpec(),
        image_key: Optional[str] = None,
        agent_pos_key: str = "agent_pos",
        action_key: str = "action",
        in_memory: bool = False, # True -> load all data into RAM
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.window = window
        self.device = device

        root = zarr.open(zarr_path, mode="r")
        data = root["data"] # data group
        meta = root["meta"] # metadata group

        # episode boundaries
        episode_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64) # convert to numpy array

        # choose image key (some replays may store "image" or "img")
        if image_key is None:
            if "image" in data:
                image_key = "image"
            elif "img" in data:
                image_key = "img"
            else:
                raise KeyError(f"Could not find image key in zarr['data']. keys={list(data.array_keys())}")

        self.keys = dict(image=image_key, agent_pos=agent_pos_key, action=action_key)

        # data arrays (either zarr arrays or numpy arrays)
        # if in_memory=True, we load all data into RAM as numpy arrays
        # else we keep zarr arrays and load slices on-the-fly (slower, but saves RAM)
        def maybe_load(arr):
            return np.asarray(arr[:]) if in_memory else arr

        self._arr_image = maybe_load(data[self.keys["image"]])
        self._arr_agent_pos = maybe_load(data[self.keys["agent_pos"]])
        self._arr_action = maybe_load(data[self.keys["action"]])

        # indices for all possible windows (with padding)
        self.indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=window.horizon,
            pad_before=window.pad_before,
            pad_after=window.pad_after,
        )

        # compute min/max stats for normalization (agent_pos, action)
        ap = np.asarray(self._arr_agent_pos[:]) if not in_memory else self._arr_agent_pos
        ac = np.asarray(self._arr_action[:]) if not in_memory else self._arr_action

        self.normalizer = MinMaxNormalizer(
            stats={
                "agent_pos": MinMaxNormalizer.compute_stats(ap),
                "action": MinMaxNormalizer.compute_stats(ac),
            }
        )

        # expose dims
        self.action_dim = ac.shape[-1] # number of action dimensions: 2
        # images assumed (N, H, W, C) or (N, C, H, W); we handle both
        img0 = self._get_image_slice(0, 1)
        self.image_shape_chw = img0.shape[1:]  # (C,H,W)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _get_image_slice(self, start: int, end: int) -> np.ndarray:
        """
        Returns images in (N, C, H, W) float32 in the range [0,1]
        """
        imgs = self._arr_image[start:end]
        imgs = np.asarray(imgs)  # if zarr array, this loads the slice

        # common layouts:
        # 1) (T, H, W, C) uint8
        # 2) (T, C, H, W) uint8
        if imgs.ndim != 4:
            raise ValueError(f"Expected 4D images, got shape={imgs.shape}")

        if imgs.shape[-1] == 3:  # THWC
            imgs = np.transpose(imgs, (0, 3, 1, 2))  # -> TCHW
        elif imgs.shape[1] == 3:
            pass  # already TCHW
        else:
            raise ValueError(f"Cannot infer channel dimension from shape={imgs.shape}")

        imgs = imgs.astype(np.float32)
        if imgs.max() > 1.0:
            imgs = imgs / 255.0
        return imgs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # shape (num_samples, 4) -> each row: [buffer_start, buffer_end, sample_start, sample_end]
        b0, b1, s0, s1 = self.indices[idx].tolist()

        # slice/pad arrays
        sample_np = sample_sequence(
            data={
                "agent_pos": np.asarray(self._arr_agent_pos[b0:b1]),
                "action": np.asarray(self._arr_action[b0:b1]),
                # images handled separately (need transpose)
            },
            sequence_length=self.window.horizon,
            buffer_start=0,            # already sliced above
            buffer_end=(b1 - b0),
            sample_start=s0,
            sample_end=s1,
        )

        # images: slice then pad manually (because transpose + float conversion)
        imgs = self._get_image_slice(b0, b1)  # (t, C, H, W)
        if (s0 != 0) or (s1 != self.window.horizon):
            padded = np.zeros((self.window.horizon,) + imgs.shape[1:], dtype=imgs.dtype)
            if s0 > 0:
                padded[:s0] = imgs[0]
            if s1 < self.window.horizon:
                padded[s1:] = imgs[-1]
            padded[s0:s1] = imgs
            imgs = padded

        # keep only last To observations
        To = self.window.n_obs_steps
        obs_image = imgs[:To]
        obs_agent_pos = sample_np["agent_pos"][:To]
        action = sample_np["action"]  # full horizon H

        # normalize low-dim
        obs_agent_pos = self.normalizer.normalize_np("agent_pos", obs_agent_pos)
        action = self.normalizer.normalize_np("action", action)

        # to torch
        out = {
            "obs_image": torch.from_numpy(obs_image),            # (To,C,H,W) in [0,1]
            "obs_agent_pos": torch.from_numpy(obs_agent_pos),    # (To,2) in [-1,1]
            "action": torch.from_numpy(action),                  # (H,2) in [-1,1]
        }
        if self.device is not None:
            out = {k: v.to(self.device) for k, v in out.items()}
        return out
