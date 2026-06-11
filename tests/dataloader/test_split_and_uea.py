import numpy as np
import pandas as pd
import pytest


def test_sliding_window_rejects_negative_derived_test_ratio():
    from torch_timeseries.dataloader.sliding_window_ts import SlidingWindowTS

    with pytest.raises(ValueError, match="Split ratios"):
        SlidingWindowTS(
            dataset=object(),
            scaler=object(),
            train_ratio=0.9,
            val_ratio=0.2,
        )


def test_uea_validation_comes_from_train_not_test():
    from torch_timeseries.dataloader.uea import UEAClassification
    from torch_timeseries.scaler import StandardScaler

    class FakeUEA:
        num_classes = 2

        def __init__(self):
            train_ids = np.repeat(np.arange(10), 3)
            test_ids = np.repeat(np.arange(4), 3)
            self.train_df = pd.DataFrame(
                {"a": np.arange(30, dtype=float), "b": np.arange(30, 60, dtype=float)},
                index=train_ids,
            )
            self.test_df = pd.DataFrame(
                {"a": np.arange(12, dtype=float), "b": np.arange(12, 24, dtype=float)},
                index=test_ids,
            )
            self.train_labels = pd.DataFrame([i % 2 for i in range(10)])
            self.test_labels = pd.DataFrame([i % 2 for i in range(4)])
            self.train_features_data = self.train_df.values

    loader = UEAClassification(
        FakeUEA(),
        StandardScaler(),
        val_ratio=0.2,
        batch_size=2,
        num_worker=0,
    )

    assert loader.val_dataset.flag == "train"
    assert loader.test_dataset.flag == "test"
    assert set(loader.val_dataset.indexes).isdisjoint(set(loader.train_dataset.indexes))
    assert set(loader.val_dataset.indexes) != set(loader.test_dataset.indexes)


def _make_toy_subset(n=300, num_features=5):
    """Helper: returns a TimeseriesSubset with named DataFrame columns."""
    import numpy as np
    import pandas as pd
    from torch_timeseries.core import TimeSeriesDataset, Freq, TimeseriesSubset

    _n = n
    _nf = num_features

    class _Toy(TimeSeriesDataset):
        name = "toy"
        freq = Freq.hours

        def download(self): pass

        def _load(self):
            self.num_features = _nf
            rng = np.random.default_rng(42)
            self.df = pd.DataFrame(
                {"date": pd.date_range("2020-01-01", periods=_n, freq="h"),
                 **{f"c{i}": rng.normal(size=_n) for i in range(_nf)}}
            )
            self.dates = self.df[["date"]]
            self.data  = self.df.drop("date", axis=1).values
            self.length = _n

    ds = _Toy()
    from torch_timeseries.scaler import StandardScaler
    return TimeseriesSubset(ds, range(n)), StandardScaler()


def test_windowed_dataset_column_selection_by_index():
    from torch_timeseries.dataloader.v2.windowed import WindowedDataset
    subset, scaler = _make_toy_subset()
    scaler.fit(subset.data)

    wd = WindowedDataset(
        subset, scaler,
        window=24, horizon=1, steps=12,
        input_columns=[0, 1, 2],
        target_columns=[4],
    )
    batch = wd[0]
    assert batch.x.shape == (24, 3), batch.x.shape
    assert batch.y.shape == (12, 1), batch.y.shape


def test_windowed_dataset_column_selection_by_name():
    from torch_timeseries.dataloader.v2.windowed import WindowedDataset
    subset, scaler = _make_toy_subset()
    scaler.fit(subset.data)

    wd = WindowedDataset(
        subset, scaler,
        window=24, horizon=1, steps=12,
        input_columns=["c0", "c1", "c2"],
        target_columns=["c4"],
    )
    batch = wd[0]
    assert batch.x.shape == (24, 3), batch.x.shape
    assert batch.y.shape == (12, 1), batch.y.shape
