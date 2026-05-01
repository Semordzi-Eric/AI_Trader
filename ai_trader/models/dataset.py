"""Sliding-window dataset.

Given a (T, F) feature matrix and a target series, produces (window, F)
sequences and the next-bar (or h-bar) target. Padding is *not* used —
samples without enough history are dropped, which is the only correct option.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """In-memory windowed dataset. For very large datasets, swap for a memmapped
    or chunked variant — the shape (T, F) will fit M-of-bars in float32 easily.
    """

    def __init__(
        self,
        features: np.ndarray,         # (T, F)
        targets: np.ndarray,          # (T,)
        window_size: int,
        horizon: int = 1,
    ) -> None:
        if features.shape[0] != targets.shape[0]:
            raise ValueError("features and targets must have same length")
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        self._n = self.features.shape[0] - self.window_size - self.horizon + 1
        if self._n <= 0:
            raise ValueError(
                f"Not enough data: {self.features.shape[0]} bars, "
                f"need > {self.window_size + self.horizon}"
            )

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.window_size]
        # Target is the value `horizon` bars after the end of the window.
        y_idx = idx + self.window_size + self.horizon - 1
        y = self.targets[y_idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
