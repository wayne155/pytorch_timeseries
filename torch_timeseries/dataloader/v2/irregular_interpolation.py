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
class IrregularInterpolationConfig:
    query_rate: float = 0.2
    time_enc: int = 0
    freq: Optional[str] = None

    def __post_init__(self):
        if not (0.0 < self.query_rate < 1.0):
            raise ValueError(f"query_rate must be in (0, 1), got {self.query_rate}")


class _IrregularInterpolationDataset(Dataset):
    """Splits each sample into input observations and held-out query targets."""

    def __init__(self, dataset, scaler, indices: List[int], query_rate: float) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.indices = indices
        self.query_rate = query_rate

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> IrregularTSBatch:
        i = self.indices[idx]
        x_raw = np.array(self.dataset.samples[i], dtype=np.float32)  # (T_i, F)
        t_raw = np.array(self.dataset.times[i], dtype=np.float32)     # (T_i,)
        mask = np.array(self.dataset.masks[i], dtype=np.float32)       # (T_i, F)
        T, F = x_raw.shape

        # Deterministic hold-out seeded by sample index for reproducibility
        rng = np.random.default_rng(seed=i)
        n_query = max(1, int(T * self.query_rate))
        query_idx = np.sort(rng.choice(T, size=n_query, replace=False))
        query_set = set(query_idx.tolist())
        input_idx = np.array([j for j in range(T) if j not in query_set])

        if len(input_idx) == 0:
            input_idx = query_idx[:1]
            query_idx = query_idx[1:]

        x_in = x_raw[input_idx]
        t_in = t_raw[input_idx]
        mask_in = mask[input_idx]

        x_q = x_raw[query_idx]
        t_q = t_raw[query_idx]
        q_mask = mask[query_idx]

        # Normalize time to [0, 1] using full sample range
        t_min, t_max = t_raw.min(), t_raw.max()
        eps = 1e-8
        t_in_norm = (t_in - t_min) / (t_max - t_min + eps)
        t_q_norm = (t_q - t_min) / (t_max - t_min + eps)

        x_in_scaled = self.scaler.transform(x_in) * mask_in

        return IrregularTSBatch(
            x=torch.from_numpy(x_in_scaled),
            t=torch.from_numpy(t_in_norm),
            mask=torch.from_numpy(mask_in),
            y=torch.from_numpy(x_q * q_mask),
            t_query=torch.from_numpy(t_q_norm),
            query_mask=torch.from_numpy(q_mask),
        )


class IrregularInterpolationDataModule:
    """DataModule for irregular time-series interpolation.

    Each batch: ``IrregularTSBatch(x, t, mask, y=query_values, t_query, query_mask)``.
    Loss is computed only on ``query_mask == 1`` positions.
    """

    def __init__(
        self,
        dataset,
        scaler,
        window: IrregularInterpolationConfig = None,
        split: SplitConfig = None,
        loader: LoaderConfig = None,
    ) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.window_cfg = window or IrregularInterpolationConfig()
        self.split_cfg = split or SplitConfig()
        self.loader_cfg = loader or LoaderConfig()

        self._fit_scaler()
        self._build_datasets()
        self._build_loaders()

    def _fit_scaler(self) -> None:
        n = len(self.dataset)
        train_end = int(self.split_cfg.train * n)
        obs_list = [np.array(self.dataset.samples[i], dtype=np.float32)
                    for i in range(train_end)]
        if obs_list:
            self.scaler.fit(np.vstack(obs_list))

    def _build_datasets(self) -> None:
        n = len(self.dataset)
        train_end = int(self.split_cfg.train * n)
        test_size = int((self.split_cfg.test or 0.2) * n)
        val_end = n - test_size
        qr = self.window_cfg.query_rate

        self.train_dataset = _IrregularInterpolationDataset(
            self.dataset, self.scaler, list(range(0, train_end)), qr)
        self.val_dataset = _IrregularInterpolationDataset(
            self.dataset, self.scaler, list(range(train_end, val_end)), qr)
        self.test_dataset = _IrregularInterpolationDataset(
            self.dataset, self.scaler, list(range(val_end, n)), qr)

    def _build_loaders(self) -> None:
        lc = self.loader_cfg
        kw = dict(
            batch_size=lc.batch_size, num_workers=lc.num_workers,
            pin_memory=lc.pin_memory, collate_fn=collate_irregular,
            worker_init_fn=seed_worker,
        )
        self.train_loader = DataLoader(self.train_dataset, shuffle=lc.shuffle_train, **kw)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, **kw)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, **kw)

    @property
    def num_features(self) -> int:
        return self.dataset.num_features

    @property
    def query_rate(self) -> float:
        return self.window_cfg.query_rate
