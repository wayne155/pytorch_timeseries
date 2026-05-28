# torch_timeseries/dataloader/v2/irregular_classification.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .._seed import seed_worker
from .forecast import LoaderConfig, SplitConfig
from .irregular_batch import IrregularTSBatch, collate_irregular


@dataclass
class IrregularClassificationConfig:
    time_enc: int = 0           # 0=no calendar features
    freq: Optional[str] = None  # required if time_enc > 0


class _IrregularClassificationDataset(Dataset):
    """Maps indices into an IrregularTimeSeriesDataset, scales, normalizes time."""

    def __init__(self, dataset, scaler, indices: List[int]) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> IrregularTSBatch:
        i = self.indices[idx]
        x_raw = np.array(self.dataset.samples[i], dtype=np.float32)  # (T_i, F)
        t_raw = np.array(self.dataset.times[i], dtype=np.float32)     # (T_i,)
        mask = np.array(self.dataset.masks[i], dtype=np.float32)       # (T_i, F)
        label = int(self.dataset.labels[i])

        # Normalize time to [0, 1] per sample
        t_min, t_max = t_raw.min(), t_raw.max()
        t_norm = (t_raw - t_min) / (t_max - t_min + 1e-8)

        # Scale observed values; zero out unobserved positions afterward
        x_scaled = self.scaler.transform(x_raw) * mask

        return IrregularTSBatch(
            x=torch.from_numpy(x_scaled),
            t=torch.from_numpy(t_norm),
            mask=torch.from_numpy(mask),
            y=torch.tensor(label, dtype=torch.long),
        )


class IrregularClassificationDataModule:
    """DataModule for irregular time-series classification.

    Splits by sample index; fits scaler on training set observed values only.
    Each DataLoader returns IrregularTSBatch via collate_irregular.
    """

    def __init__(
        self,
        dataset,
        scaler,
        window: IrregularClassificationConfig = None,
        split: SplitConfig = None,
        loader: LoaderConfig = None,
    ) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.window_cfg = window or IrregularClassificationConfig()
        self.split_cfg = split or SplitConfig()
        self.loader_cfg = loader or LoaderConfig()

        self._fit_scaler()
        self._build_datasets()
        self._build_loaders()

    def _fit_scaler(self) -> None:
        n = len(self.dataset)
        train_end = int(self.split_cfg.train * n)
        obs_list = [
            np.array(self.dataset.samples[i], dtype=np.float32)
            for i in range(train_end)
        ]
        if obs_list:
            self.scaler.fit(np.vstack(obs_list))

    def _build_datasets(self) -> None:
        n = len(self.dataset)
        train_end = int(self.split_cfg.train * n)
        test_size = int((self.split_cfg.test or 0.2) * n)
        val_end = n - test_size

        self.train_dataset = _IrregularClassificationDataset(
            self.dataset, self.scaler, list(range(0, train_end)))
        self.val_dataset = _IrregularClassificationDataset(
            self.dataset, self.scaler, list(range(train_end, val_end)))
        self.test_dataset = _IrregularClassificationDataset(
            self.dataset, self.scaler, list(range(val_end, n)))

    def _build_loaders(self) -> None:
        lc = self.loader_cfg
        kw = dict(
            batch_size=lc.batch_size,
            num_workers=lc.num_workers,
            pin_memory=lc.pin_memory,
            collate_fn=collate_irregular,
            worker_init_fn=seed_worker,
        )
        self.train_loader = DataLoader(
            self.train_dataset, shuffle=lc.shuffle_train, **kw)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, **kw)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, **kw)

    @property
    def num_features(self) -> int:
        return self.dataset.num_features

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes
