# tests/dataloader/test_v2_task_modules.py
import numpy as np
import pandas as pd
import pytest
import torch

from torch_timeseries.core import TimeSeriesDataset, Freq
from torch_timeseries.scaler import StandardScaler
from torch_timeseries.dataloader.v2 import SplitConfig, LoaderConfig, TSBatch, ForecastDataModule, WindowConfig
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


from torch_timeseries.dataloader.v2.anomaly import AnomalyDataModule, AnomalyWindowConfig


class _ToyAnomaly:
    """Minimal stand-in for torch_timeseries anomaly dataset."""
    name = "toy_anomaly"
    num_features = 3

    def __init__(self, root=None):
        rng = np.random.default_rng(42)
        self.train_data = rng.normal(size=(400, 3)).astype(np.float32)
        self.test_data = rng.normal(size=(200, 3)).astype(np.float32)
        self.test_labels = (rng.random(200) > 0.8).astype(np.float32)


def test_anomaly_dm_train_batch_y_is_none():
    dm = AnomalyDataModule(
        dataset=_ToyAnomaly(),
        scaler=StandardScaler(),
        window=AnomalyWindowConfig(window=20, stride=10),
        loader=LoaderConfig(batch_size=8, num_workers=0),
    )
    batch = next(iter(dm.train_loader))
    assert isinstance(batch, TSBatch)
    assert batch.y is None


def test_anomaly_dm_train_batch_x_shape():
    dm = AnomalyDataModule(
        dataset=_ToyAnomaly(),
        scaler=StandardScaler(),
        window=AnomalyWindowConfig(window=20, stride=10),
        loader=LoaderConfig(batch_size=8, num_workers=0),
    )
    batch = next(iter(dm.train_loader))
    B = min(8, len(dm.train_dataset))
    assert batch.x.shape == (B, 20, 3)


def test_anomaly_dm_test_batch_y_is_labels():
    dm = AnomalyDataModule(
        dataset=_ToyAnomaly(),
        scaler=StandardScaler(),
        window=AnomalyWindowConfig(window=20, stride=10),
        loader=LoaderConfig(batch_size=8, num_workers=0),
    )
    batch = next(iter(dm.test_loader))
    assert batch.y is not None
    B = min(8, len(dm.test_dataset))
    assert batch.y.shape == (B, 20)   # per-timestep labels in window


from torch_timeseries.dataloader.v2.uea import UEADataModule, UEAWindowConfig


class _ToyUEA:
    """Minimal stand-in for a UEA dataset."""
    name = "toy_uea"
    num_features = 5
    num_classes = 3
    max_seq_len = 24

    def __init__(self, root=None):
        rng = np.random.default_rng(7)
        n_train, n_test, T, F = 60, 20, 24, 5
        idx_train = pd.MultiIndex.from_product(
            [range(n_train), range(T)], names=["sample_id", "timestep"]
        )
        idx_test = pd.MultiIndex.from_product(
            [range(n_train, n_train + n_test), range(T)], names=["sample_id", "timestep"]
        )
        self.train_df = pd.DataFrame(
            rng.normal(size=(n_train * T, F)), index=idx_train,
            columns=[f"c{i}" for i in range(F)]
        )
        self.test_df = pd.DataFrame(
            rng.normal(size=(n_test * T, F)), index=idx_test,
            columns=[f"c{i}" for i in range(F)]
        )
        train_label_idx = pd.Index(range(n_train), name="sample_id")
        test_label_idx = pd.Index(range(n_train, n_train + n_test), name="sample_id")
        self.train_labels = pd.Series(
            rng.integers(0, 3, size=n_train), index=train_label_idx
        )
        self.test_labels = pd.Series(
            rng.integers(0, 3, size=n_test), index=test_label_idx
        )


def test_uea_dm_train_batch_shape():
    dm = UEADataModule(
        dataset=_ToyUEA(),
        scaler=StandardScaler(),
        window=UEAWindowConfig(val_ratio=0.2),
        loader=LoaderConfig(batch_size=8, num_workers=0),
    )
    batch = next(iter(dm.train_loader))
    assert isinstance(batch, TSBatch)
    B = min(8, len(dm.train_dataset))
    assert batch.x.shape == (B, 24, 5)   # (B, T, F)
    assert batch.y.shape == (B,)          # integer class labels


def test_uea_dm_test_batch_shape():
    dm = UEADataModule(
        dataset=_ToyUEA(),
        scaler=StandardScaler(),
        window=UEAWindowConfig(val_ratio=0.2),
        loader=LoaderConfig(batch_size=8, num_workers=0),
    )
    batch = next(iter(dm.test_loader))
    B = min(8, len(dm.test_dataset))
    assert batch.x.shape == (B, 24, 5)
    assert batch.y.shape == (B,)


def test_uea_dm_has_val_loader():
    dm = UEADataModule(
        dataset=_ToyUEA(),
        scaler=StandardScaler(),
        window=UEAWindowConfig(val_ratio=0.2),
        loader=LoaderConfig(batch_size=8, num_workers=0),
    )
    assert hasattr(dm, "val_loader")
    assert len(dm.val_dataset) > 0


def test_window_config_accepts_string_time_enc():
    """WindowConfig should accept string aliases without error."""
    cfg = WindowConfig(time_enc="fourier")
    assert cfg.time_enc == "fourier"


def _toy_forecast_dm(time_enc="calendar"):
    return ForecastDataModule(
        dataset=_ToyTS(root="/tmp"),
        scaler=StandardScaler(),
        window=WindowConfig(window=24, horizon=4, time_enc=time_enc, freq="h"),
        split=SplitConfig(train=0.7, val=0.1, test=0.2),
        loader=LoaderConfig(batch_size=8, num_workers=0),
    )


def test_forecast_dm_string_time_enc_calendar():
    dm = _toy_forecast_dm(time_enc="calendar")
    batch = next(iter(dm.train_loader))
    assert batch.x_time_feature is not None


def test_forecast_dm_string_time_enc_fourier():
    dm = _toy_forecast_dm(time_enc="fourier")
    batch = next(iter(dm.train_loader))
    assert batch.x_time_feature is not None


def test_forecast_dm_int_time_enc_still_works():
    dm = _toy_forecast_dm(time_enc=1)
    batch = next(iter(dm.train_loader))
    assert batch.x_time_feature is not None
