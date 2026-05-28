from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .._seed import seed_worker
from .batch import TSBatch, collate_tsbatch
from .forecast import LoaderConfig


@dataclass
class AnomalyWindowConfig:
    window: int = 96
    stride: int = 1
    train_ratio: float = 0.8

    def __post_init__(self):
        if self.window <= 0:
            raise ValueError(f"window must be > 0, got {self.window}")
        if self.stride <= 0:
            raise ValueError(f"stride must be > 0, got {self.stride}")
        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError(f"train_ratio must be in (0, 1), got {self.train_ratio}")


class AnomalyWindowedDataset(Dataset):
    """Sliding-window dataset from a flat data array."""

    def __init__(
        self,
        data: np.ndarray,
        scaler,
        window: int = 96,
        stride: int = 1,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        # Fix 1: validate labels length against data length
        if labels is not None and len(labels) != len(data):
            raise ValueError(
                f"labels length {len(labels)} does not match data length {len(data)}."
            )
        if len(data) > 0 and len(data) < window:
            raise ValueError(
                f"Data length {len(data)} is shorter than window {window}."
            )
        self.window = window
        self.labels = labels
        self.scaled = scaler.transform(data).astype(np.float32)
        self._starts = list(range(0, len(data) - window + 1, stride))

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> TSBatch:
        s = self._starts[idx]
        e = s + self.window
        x = torch.from_numpy(self.scaled[s:e].copy())
        y = None
        if self.labels is not None:
            y = torch.from_numpy(self.labels[s:e].copy())
        return TSBatch(x=x, y=y)


class AnomalyDataModule:
    """Train/val from `dataset.train_data`; test from `dataset.test_data` with labels."""

    def __init__(
        self,
        dataset,
        scaler,
        window: AnomalyWindowConfig = None,
        loader: LoaderConfig = None,
    ) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.window_cfg = window or AnomalyWindowConfig()
        self.loader_cfg = loader or LoaderConfig()

        # Fix 2: separate scaler fit from dataset construction
        wc = self.window_cfg
        self._train_end = int(len(dataset.train_data) * wc.train_ratio)

        self._fit_scaler()
        self._build_datasets()
        self._build_loaders()

    def _fit_scaler(self) -> None:
        self.scaler.fit(self.dataset.train_data[:self._train_end])

    def _build_datasets(self) -> None:
        wc = self.window_cfg
        self.train_dataset = AnomalyWindowedDataset(
            self.dataset.train_data[:self._train_end], self.scaler,
            window=wc.window, stride=wc.stride,
        )
        self.val_dataset = AnomalyWindowedDataset(
            self.dataset.train_data[self._train_end:], self.scaler,
            window=wc.window, stride=wc.stride,
        )
        self.test_dataset = AnomalyWindowedDataset(
            self.dataset.test_data, self.scaler,
            window=wc.window, stride=wc.stride,
            labels=self.dataset.test_labels,
        )
        # Fix 3: warn when test_dataset is empty
        if len(self.test_dataset) == 0:
            warnings.warn(
                "AnomalyDataModule: test_dataset has 0 windows — "
                "test_data may be shorter than window.",
                stacklevel=2,
            )

    def _build_loaders(self) -> None:
        lc = self.loader_cfg
        kw = dict(
            batch_size=lc.batch_size, num_workers=lc.num_workers,
            pin_memory=lc.pin_memory, collate_fn=collate_tsbatch,
            worker_init_fn=seed_worker,
        )
        self.train_loader = DataLoader(self.train_dataset, shuffle=lc.shuffle_train, **kw)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, **kw)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, **kw)

    # Fix 4: add window property
    @property
    def window(self) -> int:
        return self.window_cfg.window

    @property
    def num_features(self) -> int:
        return self.dataset.num_features
