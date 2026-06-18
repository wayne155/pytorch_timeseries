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
class IrregularForecastConfig:
    obs_frac: float = 0.7   # fraction of timespan used as input context
    time_enc: int = 1
    freq: Optional[str] = None

    def __post_init__(self):
        if not (0.0 < self.obs_frac < 1.0):
            raise ValueError(f"obs_frac must be in (0, 1), got {self.obs_frac}")


class _IrregularForecastDataset(Dataset):
    """Splits each sample at obs_frac of its timespan into input + forecast targets."""

    def __init__(self, dataset, scaler, indices: List[int], obs_frac: float) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.indices = indices
        self.obs_frac = obs_frac

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> IrregularTSBatch:
        i = self.indices[idx]
        x_raw = np.array(self.dataset.samples[i], dtype=np.float32)  # (T_i, F)
        t_raw = np.array(self.dataset.times[i], dtype=np.float32)     # (T_i,)
        mask = np.array(self.dataset.masks[i], dtype=np.float32)       # (T_i, F)

        t_min, t_max = t_raw.min(), t_raw.max()
        split_t = t_min + self.obs_frac * (t_max - t_min)

        input_idx = np.where(t_raw <= split_t)[0]
        future_idx = np.where(t_raw > split_t)[0]

        # If no future points, take the last 20% as future
        if len(future_idx) == 0:
            n_future = max(1, len(t_raw) // 5)
            input_idx = np.arange(len(t_raw) - n_future)
            future_idx = np.arange(len(t_raw) - n_future, len(t_raw))

        if len(input_idx) == 0:
            input_idx = future_idx[:1]
            future_idx = future_idx[1:]

        x_in = x_raw[input_idx]
        t_in = t_raw[input_idx]
        mask_in = mask[input_idx]

        x_fut = x_raw[future_idx]
        t_fut = t_raw[future_idx]
        mask_fut = mask[future_idx]

        # Normalize time globally per sample
        eps = 1e-8
        t_in_norm = (t_in - t_min) / (t_max - t_min + eps)
        t_fut_norm = (t_fut - t_min) / (t_max - t_min + eps)

        x_in_scaled = self.scaler.transform(x_in) * mask_in

        return IrregularTSBatch(
            x=torch.from_numpy(x_in_scaled),
            t=torch.from_numpy(t_in_norm),
            mask=torch.from_numpy(mask_in),
            y=torch.from_numpy(x_fut * mask_fut),
            t_query=torch.from_numpy(t_fut_norm),
            query_mask=torch.from_numpy(mask_fut),
        )


class IrregularForecastDataModule:
    """DataModule for irregular time-series forecasting.

    Each batch: ``IrregularTSBatch(x, t, mask, y=future_values, t_query, query_mask)``.
    Observations before ``obs_frac`` of the sample's timespan are the input;
    observations after are forecast targets.
    """

    def __init__(
        self,
        dataset,
        scaler,
        window: IrregularForecastConfig = None,
        split: SplitConfig = None,
        loader: LoaderConfig = None,
    ) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.window_cfg = window or IrregularForecastConfig()
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
        of = self.window_cfg.obs_frac

        self.train_dataset = _IrregularForecastDataset(
            self.dataset, self.scaler, list(range(0, train_end)), of)
        self.val_dataset = _IrregularForecastDataset(
            self.dataset, self.scaler, list(range(train_end, val_end)), of)
        self.test_dataset = _IrregularForecastDataset(
            self.dataset, self.scaler, list(range(val_end, n)), of)

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
    def obs_frac(self) -> float:
        return self.window_cfg.obs_frac
