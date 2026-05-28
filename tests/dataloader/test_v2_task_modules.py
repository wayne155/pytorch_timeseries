# tests/dataloader/test_v2_task_modules.py
import numpy as np
import pandas as pd
import pytest
import torch

from torch_timeseries.core import TimeSeriesDataset, Freq
from torch_timeseries.scaler import StandardScaler
from torch_timeseries.dataloader.v2 import SplitConfig, LoaderConfig, TSBatch
from torch_timeseries.dataloader.v2.imputation import (
    ImputationDataModule, ImputationWindowConfig,
)


class _ToyTS(TimeSeriesDataset):
    name = "toy_ts"
    num_features = 4
    freq = Freq.hours

    def download(self): pass

    def _load(self):
        n = 500
        rng = np.random.default_rng(0)
        self.df = pd.DataFrame(
            {"date": pd.date_range("2020-01-01", periods=n, freq="h"),
             **{f"c{i}": rng.normal(size=n) for i in range(4)}}
        )
        self.dates = self.df[["date"]]
        self.data = self.df.drop("date", axis=1).values
        self.length = n


def _toy_imputation_dm(window=32, **kw):
    return ImputationDataModule(
        dataset=_ToyTS(root="/tmp"),
        scaler=StandardScaler(),
        window=ImputationWindowConfig(window=window, **kw),
        split=SplitConfig(train=0.7, val=0.1, test=0.2),
        loader=LoaderConfig(batch_size=16, num_workers=0),
    )


def test_imputation_dm_returns_tsbatch():
    dm = _toy_imputation_dm()
    batch = next(iter(dm.train_loader))
    assert isinstance(batch, TSBatch)


def test_imputation_dm_batch_shapes():
    dm = _toy_imputation_dm(window=32)
    batch = next(iter(dm.train_loader))
    B = min(16, len(dm.train_dataset))
    assert batch.x.shape == (B, 32, 4)
    assert batch.y.shape == (B, 32, 4)


def test_imputation_dm_x_y_equal_values():
    """y must equal x before masking is applied (reconstruction target)."""
    dm = _toy_imputation_dm(window=32)
    batch = next(iter(dm.train_loader))
    assert torch.allclose(batch.x, batch.y)


def test_imputation_dm_x_y_independent_tensors():
    """x and y must be separate tensors so masking x doesn't corrupt y."""
    dm = _toy_imputation_dm(window=32)
    batch = next(iter(dm.train_loader))
    assert batch.x.data_ptr() != batch.y.data_ptr()


def test_imputation_dm_has_all_loaders():
    dm = _toy_imputation_dm()
    assert hasattr(dm, "train_loader")
    assert hasattr(dm, "val_loader")
    assert hasattr(dm, "test_loader")


def test_imputation_dm_mask_ratio_accessible():
    dm = _toy_imputation_dm(mask_ratio=0.3)
    assert dm.mask_ratio == pytest.approx(0.3)


def test_imputation_window_config_rejects_bad_mask_ratio():
    with pytest.raises(ValueError, match="mask_ratio"):
        ImputationWindowConfig(mask_ratio=1.5)


def test_imputation_window_config_rejects_bad_window():
    with pytest.raises(ValueError, match="window"):
        ImputationWindowConfig(window=0)


def test_imputation_dm_empty_val_split():
    dm = ImputationDataModule(
        dataset=_ToyTS(root="/tmp"),
        scaler=StandardScaler(),
        window=ImputationWindowConfig(window=32),
        split=SplitConfig(train=0.8, val=0.0, test=0.2),
        loader=LoaderConfig(batch_size=16, num_workers=0),
    )
    assert len(dm.val_dataset) == 0
    batches = list(dm.val_loader)
    assert len(batches) == 0
