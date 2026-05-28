from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .._seed import seed_worker
from .batch import TSBatch, collate_tsbatch
from .forecast import LoaderConfig


@dataclass
class UEAWindowConfig:
    val_ratio: float = 0.2
    normalize: bool = True

    def __post_init__(self):
        if not (0.0 < self.val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in (0, 1), got {self.val_ratio}")


class UEAWindowedDataset(Dataset):
    """One sample per unique sample_id in the multi-index DataFrame."""

    def __init__(
        self,
        feature_df: pd.DataFrame,
        labels: pd.Series,
        scaler,
        sample_ids,
        scaler_fit: bool = False,
    ) -> None:
        self.sample_ids = list(sample_ids)
        if scaler_fit and len(self.sample_ids) > 0:
            scaler.fit(feature_df.loc[self.sample_ids].values)
        self.scaled_df = pd.DataFrame(
            scaler.transform(feature_df.values),
            index=feature_df.index,
            columns=feature_df.columns,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> TSBatch:
        sid = self.sample_ids[idx]
        x = torch.from_numpy(self.scaled_df.loc[sid].values.astype(np.float32))
        y = torch.tensor(int(self.labels.loc[sid]), dtype=torch.long)
        return TSBatch(x=x, y=y)


class UEADataModule:
    """Wraps a UEA dataset's pre-split train/test DataFrames."""

    def __init__(
        self,
        dataset,
        scaler,
        window: UEAWindowConfig = None,
        loader: LoaderConfig = None,
    ) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.window_cfg = window or UEAWindowConfig()
        self.loader_cfg = loader or LoaderConfig()

        self._fit_scaler()
        self._build_datasets()
        self._build_loaders()

    def _fit_scaler(self) -> None:
        train_ids = list(self.dataset.train_df.index.get_level_values("sample_id").unique())
        val_split = int(len(train_ids) * (1 - self.window_cfg.val_ratio))
        train_ids_tr = train_ids[:val_split]
        self.scaler.fit(self.dataset.train_df.loc[train_ids_tr].values)

    def _build_datasets(self) -> None:
        train_ids = list(self.dataset.train_df.index.get_level_values("sample_id").unique())
        val_split = int(len(train_ids) * (1 - self.window_cfg.val_ratio))
        train_ids_tr = train_ids[:val_split]
        train_ids_val = train_ids[val_split:]
        test_ids = list(self.dataset.test_df.index.get_level_values("sample_id").unique())

        all_df = pd.concat([self.dataset.train_df, self.dataset.test_df])
        all_labels = pd.concat([self.dataset.train_labels, self.dataset.test_labels])

        self.train_dataset = UEAWindowedDataset(all_df, all_labels, self.scaler, train_ids_tr)
        self.val_dataset = UEAWindowedDataset(all_df, all_labels, self.scaler, train_ids_val)
        self.test_dataset = UEAWindowedDataset(all_df, all_labels, self.scaler, test_ids)

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
    def num_classes(self) -> int:
        return self.dataset.num_classes
