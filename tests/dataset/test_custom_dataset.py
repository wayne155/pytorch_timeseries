"""Tests for torch_timeseries.dataset.build_dataset and CSVDataset."""
import numpy as np
import pandas as pd
import pytest

from torch_timeseries.core import Freq
from torch_timeseries.dataset.custom import build_dataset, CSVDataset


def _write_csv(path, n_rows=100, n_features=3, freq="h"):
    dates = pd.date_range("2020-01-01", periods=n_rows, freq=freq)
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {"date": dates, **{f"f{i}": rng.normal(size=n_rows) for i in range(n_features)}}
    )
    df.to_csv(path, index=False)
    return path


# ── build_dataset ─────────────────────────────────────────────────────────────

class TestBuildDataset:
    def test_returns_csv_dataset(self, tmp_path):
        p = _write_csv(tmp_path / "data.csv")
        ds = build_dataset(csv=str(p), freq=Freq.hours)
        assert isinstance(ds, CSVDataset)

    def test_num_features_inferred(self, tmp_path):
        p = _write_csv(tmp_path / "data.csv", n_features=5)
        ds = build_dataset(csv=str(p), freq=Freq.hours)
        assert ds.num_features == 5

    def test_length_inferred(self, tmp_path):
        p = _write_csv(tmp_path / "data.csv", n_rows=200)
        ds = build_dataset(csv=str(p), freq=Freq.hours)
        assert ds.length == 200

    def test_data_array_shape(self, tmp_path):
        p = _write_csv(tmp_path / "data.csv", n_rows=100, n_features=7)
        ds = build_dataset(csv=str(p), freq=Freq.hours)
        assert ds.data.shape == (100, 7)

    def test_name_defaults_to_filename(self, tmp_path):
        p = _write_csv(tmp_path / "myseries.csv")
        ds = build_dataset(csv=str(p), freq=Freq.hours)
        assert ds.name == "myseries"

    def test_custom_name(self, tmp_path):
        p = _write_csv(tmp_path / "data.csv")
        ds = build_dataset(csv=str(p), freq=Freq.hours, name="MyDataset")
        assert ds.name == "MyDataset"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_dataset(csv=str(tmp_path / "nonexistent.csv"), freq=Freq.hours)

    def test_custom_date_column(self, tmp_path):
        dates = pd.date_range("2020-01-01", periods=50, freq="h")
        df = pd.DataFrame({"timestamp": dates, "val": range(50)})
        p = tmp_path / "custom_col.csv"
        df.to_csv(p, index=False)
        ds = build_dataset(csv=str(p), freq=Freq.hours, date_column="timestamp")
        assert ds.num_features == 1
        assert ds.length == 50

    def test_data_no_nan(self, tmp_path):
        p = _write_csv(tmp_path / "data.csv")
        ds = build_dataset(csv=str(p), freq=Freq.hours)
        assert not np.isnan(ds.data).any()

    def test_time_index_length(self, tmp_path):
        p = _write_csv(tmp_path / "data.csv", n_rows=80)
        ds = build_dataset(csv=str(p), freq=Freq.hours)
        assert len(ds.time_index) == 80

    def test_daily_freq(self, tmp_path):
        p = _write_csv(tmp_path / "daily.csv", freq="D")
        ds = build_dataset(csv=str(p), freq=Freq.days)
        assert ds.freq == Freq.days
