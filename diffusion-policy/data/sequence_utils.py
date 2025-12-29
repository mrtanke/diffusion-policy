# diffusion-policy/data/sequence_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class SequenceIndex:
    buffer_start: int
    buffer_end: int
    sample_start: int
    sample_end: int


def create_sample_indices(
    episode_ends: np.ndarray, # shape (num_episodes,) [5, 10] -> episodes1: 0-4, episodes2: 5-9
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
) -> np.ndarray:
    """
    Create indices for extracting fixed-length sequences from a replay buffer
    with padding support (repeat boundary frames when out of range).

    episode_ends: array of shape (num_episodes, episode_end) where each entry is
                 one-past-the-last index in the concatenated buffer.
    """
    indices = []
    for ep_i in range(len(episode_ends)):
        ep_start = 0 if ep_i == 0 else int(episode_ends[ep_i - 1])
        ep_end = int(episode_ends[ep_i])
        ep_len = ep_end - ep_start

        # allow starts earlier (pad_before) and later (pad_after)
        min_start = -pad_before
        max_start = ep_len - sequence_length + pad_after

        for start in range(min_start, max_start + 1): # from min_start to max_start inclusive
            buffer_start = max(start, 0) + ep_start
            buffer_end = min(start + sequence_length, ep_len) + ep_start

            start_offset = buffer_start - (start + ep_start)
            end_offset = (start + sequence_length + ep_start) - buffer_end

            sample_start = start_offset
            sample_end = sequence_length - end_offset

            indices.append([buffer_start, buffer_end, sample_start, sample_end]) # buffer_end is exclusive

    # shape (num_samples, 4) -> each row: [buffer_start, buffer_end, sample_start, sample_end]
    return np.asarray(indices, dtype=np.int64) 


def sample_sequence(
    data: Dict[str, np.ndarray], # A dictionary containing replay buffer (e.g., {"obs": ..., "action": ...}) 
    sequence_length: int,
    buffer_start: int,
    buffer_end: int,
    sample_start: int,
    sample_end: int,
) -> Dict[str, np.ndarray]:
    """
    Slice [buffer_start:buffer_end] then pad into a fixed-length array
    by repeating the boundary element when needed.
    """
    out: Dict[str, np.ndarray] = {}

    for key, arr in data.items():
        sliced = arr[buffer_start:buffer_end]
        if (sample_start == 0) and (sample_end == sequence_length):
            out[key] = sliced
            continue

        padded = np.zeros((sequence_length,) + arr.shape[1:], dtype=arr.dtype)
        # left pad with first
        if sample_start > 0:
            padded[:sample_start] = sliced[0]
        # right pad with last
        if sample_end < sequence_length:
            padded[sample_end:] = sliced[-1]
        # fill in the middle
        padded[sample_start:sample_end] = sliced

        out[key] = padded

    return out
