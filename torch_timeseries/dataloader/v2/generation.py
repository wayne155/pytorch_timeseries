"""Windowed dataloader for time series generation (no x/y split)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ...scaler import Scaler
from .batch import TSBatch, collate_tsbatch
from .loader import LoaderConfig
from .split import SplitConfig, default_split_config


@dataclass
class GenerationWindowConfig:
    seq_len: int = 96
    stride: int = 1
    fast_eval: bool = False


class _GenWindowedDataset(Dataset):
    """Sliding-window dataset for generation tasks."""

    def __init__(self, data: np.ndarray, seq_len: int, stride: int = 1,
                 labels: Optional[np.ndarray] = None):
        self.data = data          # (T, C) float32
        self.seq_len = seq_len
        self.labels = labels      # (N,) int or None
        T = data.shape[0]
        self.indices = list(range(0, T - seq_len + 1, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> TSBatch:
        start = self.indices[idx]
        x = torch.tensor(self.data[start: start + self.seq_len], dtype=torch.float32)
        y = (torch.tensor(int(self.labels[idx]), dtype=torch.long)
             if self.labels is not None else None)
        x_index = torch.tensor(start, dtype=torch.long)
        return TSBatch(x=x, y=y, x_index=x_index)


class GenerationDataModule:
    """DataModule for time series generation.

    Each batch is a TSBatch with x = (B, seq_len, C) (scaled) and
    y = (B,) class labels or None for unconditional generation.
    """

    def __init__(
        self,
        dataset,
        scaler,
        window: GenerationWindowConfig,
        split: Optional[SplitConfig] = None,
        loader: LoaderConfig = None,
    ) -> None:
        if loader is None:
            loader = LoaderConfig()
        self.dataset = dataset
        self.scaler = scaler
        self.window = window
        self._split = split or default_split_config(dataset)
        self.loader = loader
        self._build()

    def _build(self) -> None:
        data = self.dataset.data  # (T, C) numpy
        T = data.shape[0]
        split = self._split

        if split.borders is not None:
            train_end, val_end, test_end = split.borders
        else:
            train_end = int(T * split.train)
            val_frac = split.val if split.val is not None else 0.1
            val_end = train_end + int(T * val_frac)
            test_end = T

        train_data = data[:train_end]
        test_data = data[val_end:test_end]

        self.scaler.fit(train_data)
        train_scaled = self.scaler.transform(train_data).astype(np.float32)
        test_scaled = self.scaler.transform(test_data).astype(np.float32)

        eval_stride = self.window.seq_len if self.window.fast_eval else 1

        self._train_ds = _GenWindowedDataset(
            train_scaled, self.window.seq_len, stride=self.window.stride
        )
        self._test_ds = _GenWindowedDataset(
            test_scaled, self.window.seq_len, stride=eval_stride
        )

    @property
    def num_features(self) -> int:
        return int(self.dataset.num_features)

    @property
    def train_loader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.loader.batch_size,
            shuffle=self.loader.shuffle_train,
            num_workers=self.loader.num_workers,
            pin_memory=self.loader.pin_memory,
            collate_fn=collate_tsbatch,
        )

    @property
    def test_loader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            batch_size=self.loader.batch_size,
            shuffle=False,
            num_workers=self.loader.num_workers,
            pin_memory=self.loader.pin_memory,
            collate_fn=collate_tsbatch,
        )
