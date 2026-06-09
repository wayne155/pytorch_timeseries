# torch_timeseries/dataloader/v2/imputation.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from torch_timeseries.core import TimeSeriesDataset, TimeseriesSubset

from .._seed import seed_worker
from .._split import resolve_split_ratios
from .batch import TSBatch, Time, TimeEncConfig, collate_tsbatch
from .windowed import _slice_time
from .forecast import SplitConfig, LoaderConfig


@dataclass
class ImputationWindowConfig:
    window: int = 96
    mask_ratio: float = 0.25
    stride: int = 1
    time_enc_cfg: TimeEncConfig = field(default_factory=TimeEncConfig)

    def __post_init__(self):
        if self.window <= 0:
            raise ValueError(f"window must be > 0, got {self.window}")
        if self.stride <= 0:
            raise ValueError(f"stride must be > 0, got {self.stride}")
        if not (0.0 < self.mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in (0, 1), got {self.mask_ratio}")


class ImputationWindowedDataset(Dataset):
    """Sliding-window dataset for masked reconstruction.

    Returns TSBatch where x == y (both are copies of the same scaled window).
    The calling experiment applies a random mask to x in _prepare_batch.
    """

    def __init__(
        self,
        subset: TimeseriesSubset,
        scaler,
        window: int = 96,
        stride: int = 1,
        time_enc_cfg: TimeEncConfig = None,
    ) -> None:
        if 0 < len(subset) < window:
            raise ValueError(
                f"Subset length {len(subset)} is shorter than window {window}."
            )
        self.window = window
        _cfg = time_enc_cfg or TimeEncConfig()
        self.scaled = scaler.transform(subset.data).astype(np.float32)
        self.raw = subset.data.astype(np.float32)
        if subset.dates is not None:
            try:
                self._time_feature = _cfg.encode(subset.dates, subset.freq)
            except Exception:
                self._time_feature = None
            try:
                self._time = Time.from_dates(subset.dates)
            except Exception:
                self._time = None
        else:
            self._time_feature = None
            self._time = None
        self._starts = list(range(0, len(subset) - window + 1, stride))

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> TSBatch:
        s = self._starts[idx]
        e = s + self.window
        sl = slice(s, e)
        x = torch.from_numpy(self.scaled[sl].copy())
        tf = self._time_feature
        return TSBatch(
            x=x,
            y=x.clone(),
            x_raw=torch.from_numpy(self.raw[sl].copy()),
            x_time_feature=tf[sl] if tf is not None else None,
            x_time=_slice_time(self._time, sl),
        )


class ImputationDataModule:
    """Wires dataset -> scaler -> split -> ImputationWindowedDataset -> DataLoader."""

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        scaler,
        window: ImputationWindowConfig = None,
        split: SplitConfig = None,
        loader: LoaderConfig = None,
        scale_in_train: bool = True,
    ) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.window_cfg = window or ImputationWindowConfig()
        self.split_cfg = split or SplitConfig()
        self.loader_cfg = loader or LoaderConfig()
        self.scale_in_train = scale_in_train

        self._build_subsets()
        self._fit_scaler()
        self._build_datasets()
        self._build_loaders()

    def _build_subsets(self) -> None:
        train, val, test = resolve_split_ratios(
            train_ratio=self.split_cfg.train,
            val_ratio=self.split_cfg.val,
            test_ratio=self.split_cfg.test,
        )
        n = len(self.dataset)
        train_size = int(train * n)
        val_size = int(val * n)
        pad = self.window_cfg.window - 1 if self.split_cfg.uniform_eval else 0
        idx = range(n)
        self.train_subset = TimeseriesSubset(self.dataset, idx[:train_size])
        if val_size == 0:
            val_indices = idx[train_size:train_size]  # empty
        else:
            val_indices = (
                idx[train_size - pad: train_size + val_size]
                if self.split_cfg.uniform_eval
                else idx[train_size: train_size + val_size]
            )
        self.val_subset = TimeseriesSubset(self.dataset, val_indices)
        test_indices = (
            idx[train_size + val_size - pad:]
            if (self.split_cfg.uniform_eval and val_size > 0)
            else idx[train_size + val_size:]
        )
        self.test_subset = TimeseriesSubset(self.dataset, test_indices)

    def _fit_scaler(self) -> None:
        data = self.train_subset.data if self.scale_in_train else self.dataset.data
        self.scaler.fit(data)

    def _build_datasets(self) -> None:
        wc = self.window_cfg
        kw = dict(window=wc.window, stride=wc.stride, time_enc_cfg=wc.time_enc_cfg)
        self.train_dataset = ImputationWindowedDataset(self.train_subset, self.scaler, **kw)
        self.val_dataset = ImputationWindowedDataset(self.val_subset, self.scaler, **kw)
        self.test_dataset = ImputationWindowedDataset(self.test_subset, self.scaler, **kw)

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

    @property
    def num_features(self) -> int:
        return self.dataset.num_features

    @property
    def window(self) -> int:
        return self.window_cfg.window

    @property
    def mask_ratio(self) -> float:
        return self.window_cfg.mask_ratio
