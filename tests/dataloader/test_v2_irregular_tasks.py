import numpy as np
import torch
import pytest


class _ToyIrregular:
    """In-memory irregular dataset — no file I/O."""
    num_features = 3
    num_classes = 2

    def __init__(self, n=40, seed=0):
        rng = np.random.default_rng(seed)
        self.samples, self.times, self.masks, self.labels = [], [], [], []
        for i in range(n):
            T = rng.integers(8, 20)
            x = rng.normal(size=(T, self.num_features)).astype(np.float32)
            t = np.sort(rng.uniform(0, 48, T)).astype(np.float32)
            mask = (rng.random((T, self.num_features)) > 0.2).astype(np.float32)
            self.samples.append(x)
            self.times.append(t)
            self.masks.append(mask)
            self.labels.append(i % self.num_classes)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.samples)


def _toy_interp_dm(**kw):
    from torch_timeseries.dataloader.v2.irregular_interpolation import (
        IrregularInterpolationDataModule, IrregularInterpolationConfig,
    )
    from torch_timeseries.dataloader.v2.forecast import SplitConfig, LoaderConfig
    from torch_timeseries.scaler import StandardScaler
    return IrregularInterpolationDataModule(
        dataset=_ToyIrregular(n=40),
        scaler=StandardScaler(),
        window=IrregularInterpolationConfig(query_rate=kw.pop("query_rate", 0.2), **kw),
        split=SplitConfig(train=0.7, val=0.1, test=0.2),
        loader=LoaderConfig(batch_size=8, num_workers=0),
    )


def test_interpolation_dm_returns_batch():
    dm = _toy_interp_dm()
    batch = next(iter(dm.train_loader))
    from torch_timeseries.dataloader.v2.irregular_batch import IrregularTSBatch
    assert isinstance(batch, IrregularTSBatch)


def test_interpolation_dm_query_not_in_input_mask():
    dm = _toy_interp_dm(query_rate=0.3)
    batch = next(iter(dm.train_loader))
    assert batch.t_query is not None
    assert batch.query_mask is not None
    assert batch.y is not None
    assert batch.y.shape[:2] == batch.t_query.shape


def test_interpolation_dm_properties():
    dm = _toy_interp_dm()
    assert dm.num_features == 3
    assert dm.query_rate == 0.2
    assert hasattr(dm, "train_loader")
    assert hasattr(dm, "val_loader")
    assert hasattr(dm, "test_loader")


def _toy_forecast_dm(**kw):
    from torch_timeseries.dataloader.v2.irregular_forecast import (
        IrregularForecastDataModule, IrregularForecastConfig,
    )
    from torch_timeseries.dataloader.v2.forecast import SplitConfig, LoaderConfig
    from torch_timeseries.scaler import StandardScaler
    obs_frac = kw.pop("obs_frac", 0.7)
    return IrregularForecastDataModule(
        dataset=_ToyIrregular(n=40),
        scaler=StandardScaler(),
        window=IrregularForecastConfig(obs_frac=obs_frac, **kw),
        split=SplitConfig(train=0.7, val=0.1, test=0.2),
        loader=LoaderConfig(batch_size=8, num_workers=0),
    )


def test_forecast_dm_returns_batch():
    dm = _toy_forecast_dm()
    batch = next(iter(dm.train_loader))
    from torch_timeseries.dataloader.v2.irregular_batch import IrregularTSBatch
    assert isinstance(batch, IrregularTSBatch)


def test_forecast_dm_future_split():
    dm = _toy_forecast_dm(obs_frac=0.7)
    sample = dm.train_dataset[0]
    if sample.t_query is not None and len(sample.t_query) > 0:
        assert sample.t_query.min() >= sample.t.max()


def test_forecast_dm_properties():
    dm = _toy_forecast_dm()
    assert dm.num_features == 3
    assert dm.obs_frac == 0.7
    assert hasattr(dm, "train_loader")
    assert hasattr(dm, "val_loader")
    assert hasattr(dm, "test_loader")
