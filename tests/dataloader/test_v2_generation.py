# tests/dataloader/test_v2_generation.py
import numpy as np
import pytest
import pandas as pd
from torch_timeseries.dataloader.v2.generation import (
    GenerationWindowConfig, GenerationDataModule,
)
from torch_timeseries.dataloader.v2 import LoaderConfig, SplitConfig
from torch_timeseries.dataloader.v2.batch import TSBatch
from torch_timeseries.scaler import StandardScaler


class _ToyDataset:
    name = "__toy_gen__"
    freq = "h"
    def __init__(self, T=200, C=3):
        self.data = np.random.randn(T, C).astype(np.float32)
        self.num_features = C
        self.length = T
        self.dates = pd.DataFrame(
            {"date": pd.date_range("2020-01-01", periods=T, freq="h")}
        )


def _make_dm(seq_len=16, T=100, C=3):
    ds = _ToyDataset(T=T, C=C)
    return GenerationDataModule(
        dataset=ds,
        scaler=StandardScaler(),
        window=GenerationWindowConfig(seq_len=seq_len),
        split=SplitConfig(train=0.7, val=0.1, test=0.2),
        loader=LoaderConfig(batch_size=8, num_workers=0),
    )


def test_train_batch_is_tsbatch():
    dm = _make_dm()
    batch = next(iter(dm.train_loader))
    assert isinstance(batch, TSBatch)


def test_train_x_shape():
    dm = _make_dm(seq_len=16, T=100, C=3)
    batch = next(iter(dm.train_loader))
    B, T, C = batch.x.shape
    assert T == 16
    assert C == 3


def test_test_loader_exists():
    dm = _make_dm()
    batch = next(iter(dm.test_loader))
    assert batch.x is not None


def test_y_is_none_unconditional():
    dm = _make_dm()
    batch = next(iter(dm.train_loader))
    assert batch.y is None


def test_num_features():
    dm = _make_dm(C=5)
    assert dm.num_features == 5


def test_generation_in_task_suffixes():
    from torch_timeseries.experiments.registry import TASK_SUFFIXES
    assert "Generation" in TASK_SUFFIXES
