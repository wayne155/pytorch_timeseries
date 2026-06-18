import pytest
import numpy as np


def test_mimic_raises_if_directory_missing(tmp_path):
    from torch_timeseries.dataset.irregular.mimic import MIMIC
    with pytest.raises(FileNotFoundError, match="MIMIC"):
        MIMIC(data_dir=str(tmp_path / "nonexistent"))


def test_mimic_loads_from_fake_directory(tmp_path):
    import pandas as pd
    from torch_timeseries.dataset.irregular.mimic import MIMIC

    mimic_dir = tmp_path / "mimic"
    mimic_dir.mkdir()
    df = pd.DataFrame({
        "subject_id": [1, 1, 2, 2],
        "hours_from_admit": [0.0, 2.0, 0.0, 1.0],
        "HR": [80.0, 85.0, 72.0, float("nan")],
        "MAP": [float("nan"), 75.0, 65.0, 70.0],
        "label": [0, 0, 1, 1],
    })
    df.to_csv(mimic_dir / "mimic_processed.csv", index=False)

    ds = MIMIC(data_dir=str(mimic_dir))
    assert len(ds) == 2
    assert ds.num_features == 2
    assert ds.num_classes == 2
    assert ds.labels is not None
    assert ds.samples[0].shape[1] == 2
    assert ds.masks[0].shape == ds.samples[0].shape


def _make_fake_uea():
    import pandas as pd

    class FakeUEA:
        num_features = 2
        num_classes = 2

        def __init__(self):
            train_ids = np.repeat(np.arange(6), 20)
            test_ids = np.repeat(np.arange(2), 20)
            vals = np.random.randn(len(train_ids), 2)
            self.train_df = pd.DataFrame(
                vals, columns=["a", "b"],
                index=pd.MultiIndex.from_arrays(
                    [train_ids, np.tile(np.arange(20), 6)],
                    names=["sample_id", "time_step"]
                )
            )
            test_vals = np.random.randn(len(test_ids), 2)
            self.test_df = pd.DataFrame(
                test_vals, columns=["a", "b"],
                index=pd.MultiIndex.from_arrays(
                    [test_ids, np.tile(np.arange(20), 2)],
                    names=["sample_id", "time_step"]
                )
            )
            self.train_labels = pd.Series(
                [i % 2 for i in range(6)], index=np.arange(6), name="label")
            self.test_labels = pd.Series(
                [0, 1], index=np.arange(2), name="label")

    return FakeUEA()


def test_uea_irregular_shapes():
    from torch_timeseries.dataset.irregular.uea_irregular import UEAIrregular

    ds = UEAIrregular(_make_fake_uea(), drop_rate=0.3, seed=42)

    assert len(ds) == 8
    assert ds.num_features == 2
    assert ds.num_classes == 2
    for i in range(len(ds)):
        T_i = len(ds.times[i])
        assert T_i > 0
        assert ds.samples[i].shape == (T_i, 2)
        assert ds.masks[i].shape == (T_i, 2)
        assert ds.masks[i].sum() > 0


def test_irregular_wrapper_shapes(tmp_path):
    import pandas as pd
    from torch_timeseries.core import TimeSeriesDataset, Freq
    from torch_timeseries.dataset.irregular.wrapper import IrregularWrapper

    class FakeTS(TimeSeriesDataset):
        name = "fake"
        freq = Freq.hours
        def download(self): pass
        def _load(self):
            n = 100
            self.num_features = 3
            rng = np.random.default_rng(0)
            self.df = pd.DataFrame(
                {"date": pd.date_range("2020-01-01", periods=n, freq="h"),
                 **{f"c{i}": rng.normal(size=n) for i in range(3)}}
            )
            self.dates = self.df[["date"]]
            self.data = self.df.drop("date", axis=1).values
            self.length = n

    ds = IrregularWrapper(FakeTS(root=str(tmp_path)), window=24, drop_rate=0.4, seed=0)

    assert ds.num_features == 3
    assert len(ds) > 0
    for i in range(min(5, len(ds))):
        T_i = len(ds.times[i])
        assert T_i > 0
        assert T_i <= 24
        assert ds.samples[i].shape == (T_i, 3)
        assert ds.masks[i].shape == (T_i, 3)
    assert ds.labels is None
    assert ds.num_classes == 0
