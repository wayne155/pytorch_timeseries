"""Smoke test for ``dataloader.v2``: builds a synthetic ``TimeSeriesDataset``
(no network) and asserts shapes for both stride=1 and stride=total_len.
"""
import numpy as np
import pandas as pd
import torch

from torch_timeseries.core import TimeSeriesDataset, Freq
from torch_timeseries.scaler import StandardScaler
from torch_timeseries.dataloader.v2 import (
    ForecastDataModule,
    WindowConfig,
    SplitConfig,
    LoaderConfig,
    TSBatch,
)


class _ToyTS(TimeSeriesDataset):
    name = "toy_ts"
    num_features = 3
    freq = Freq.hours

    def download(self):  # no-op
        pass

    def _load(self):
        n = 1000
        rng = np.random.default_rng(0)
        self.df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=n, freq="h"),
                "a": rng.normal(size=n),
                "b": rng.normal(size=n),
                "c": rng.normal(size=n),
            }
        )
        self.dates = self.df[["date"]]
        self.data = self.df.drop("date", axis=1).values
        self.length = n


def _check(ds_name, dm, stride, *, expect_overlapping):
    b = next(iter(dm.train_loader))
    assert isinstance(b, TSBatch), f"[{ds_name}] expected TSBatch"
    n_train = len(dm.train_dataset)
    bs = min(32, n_train)
    assert b.x.shape == (bs, 96, 3), f"[{ds_name}] x shape {b.x.shape}"
    assert b.y.shape == (bs, 24, 3), f"[{ds_name}] y shape {b.y.shape}"
    assert b.x_raw is not None and b.y_raw is not None
    assert b.x_time is not None and b.y_time is not None
    assert b.x_index is None and b.y_index is None

    train_len = len(dm.train_subset)
    total_len = 96 + 1 + 24 - 1
    expected = (train_len - total_len) // stride + 1
    assert n_train == expected, f"[{ds_name}] expected {expected} starts, got {n_train}"

    if expect_overlapping:
        assert n_train > 100, f"[{ds_name}] sliding should yield many windows"
    else:
        assert n_train * stride <= train_len, f"[{ds_name}] non-overlap broken"
    print(f"[{ds_name}] ok  starts={n_train}  x={tuple(b.x.shape)}  y={tuple(b.y.shape)}")


def main() -> None:
    ds = _ToyTS()
    common = dict(
        dataset=ds,
        scaler=StandardScaler(),
        split=SplitConfig(train=0.7, val=0.1, test=0.2),
        loader=LoaderConfig(batch_size=32, num_workers=0),
    )

    dm_overlap = ForecastDataModule(
        window=WindowConfig(window=96, horizon=1, steps=24, stride=1),
        **common,
        # NOTE: scaler is shared across the two DMs below, so we need a fresh one.
    )
    _check("stride=1", dm_overlap, stride=1, expect_overlapping=True)

    total_len = 96 + 1 + 24 - 1
    common2 = dict(common)
    common2["scaler"] = StandardScaler()
    dm_nonoverlap = ForecastDataModule(
        window=WindowConfig(window=96, horizon=1, steps=24, stride=total_len),
        **common2,
    )
    _check("stride=total", dm_nonoverlap, stride=total_len, expect_overlapping=False)

    b = next(iter(dm_overlap.train_loader))
    x, y, x_time, y_time = b.as_tuple(("x", "y", "x_time", "y_time"))
    assert torch.equal(x, b.x)
    print("[as_tuple] ok")

    print("ALL OK")


if __name__ == "__main__":
    main()
