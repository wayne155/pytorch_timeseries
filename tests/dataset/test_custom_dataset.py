"""Tests for torch_timeseries.dataset.build_dataset, CSVDataset, DataFrameDataset."""
import numpy as np
import pandas as pd
import pytest

from torch_timeseries.core import Freq
from torch_timeseries.dataset.custom import (
    build_dataset, CSVDataset, DataFrameDataset, from_dataframe,
)


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


# ── DataFrameDataset / from_dataframe ────────────────────────────────────────

def _make_df(n_rows=100, n_features=3, freq="h", date_column="date"):
    dates = pd.date_range("2020-01-01", periods=n_rows, freq=freq)
    rng = np.random.default_rng(0)
    cols = {date_column: dates}
    cols.update({f"f{i}": rng.normal(size=n_rows) for i in range(n_features)})
    return pd.DataFrame(cols)


class TestDataFrameDataset:
    def test_returns_dataframe_dataset(self):
        df = _make_df()
        ds = from_dataframe(df, freq=Freq.hours)
        assert isinstance(ds, DataFrameDataset)

    def test_num_features_inferred(self):
        df = _make_df(n_features=6)
        ds = from_dataframe(df, freq=Freq.hours)
        assert ds.num_features == 6

    def test_length_inferred(self):
        df = _make_df(n_rows=150)
        ds = from_dataframe(df, freq=Freq.hours)
        assert ds.length == 150

    def test_data_shape(self):
        df = _make_df(n_rows=80, n_features=4)
        ds = from_dataframe(df, freq=Freq.hours)
        assert ds.data.shape == (80, 4)

    def test_data_is_float32(self):
        ds = from_dataframe(_make_df(), freq=Freq.hours)
        assert ds.data.dtype == np.float32

    def test_name_defaults_to_dataframe(self):
        ds = from_dataframe(_make_df(), freq=Freq.hours)
        assert ds.name == "dataframe"

    def test_custom_name(self):
        ds = from_dataframe(_make_df(), freq=Freq.hours, name="MySensors")
        assert ds.name == "MySensors"

    def test_no_nan(self):
        ds = from_dataframe(_make_df(), freq=Freq.hours)
        assert not np.isnan(ds.data).any()

    def test_time_index_length(self):
        df = _make_df(n_rows=60)
        ds = from_dataframe(df, freq=Freq.hours)
        assert len(ds.time_index) == 60

    def test_custom_date_column(self):
        df = _make_df(date_column="timestamp")
        ds = from_dataframe(df, freq=Freq.hours, date_column="timestamp")
        assert ds.num_features == 3

    def test_datetime_index(self):
        """DataFrame with DatetimeIndex (no explicit date column)."""
        df = pd.DataFrame(
            {"f0": np.arange(50.0), "f1": np.arange(50.0) * 2},
            index=pd.date_range("2021-01-01", periods=50, freq="h"),
        )
        ds = from_dataframe(df, freq=Freq.hours)
        assert ds.length == 50
        assert ds.num_features == 2

    def test_string_dates_parsed(self):
        """Date column given as strings should be parsed to datetime."""
        df = pd.DataFrame(
            {"date": [f"2020-01-0{i+1}" for i in range(5)],
             "val": np.arange(5.0)}
        )
        ds = from_dataframe(df, freq=Freq.days)
        assert ds.length == 5

    def test_missing_date_column_raises(self):
        df = pd.DataFrame({"f0": [1.0, 2.0, 3.0], "f1": [4.0, 5.0, 6.0]})
        with pytest.raises(ValueError, match="date"):
            from_dataframe(df, freq=Freq.hours)

    def test_freq_stored(self):
        ds = from_dataframe(_make_df(freq="D"), freq=Freq.days)
        assert ds.freq == Freq.days

    def test_source_df_not_mutated(self):
        df = _make_df()
        original_cols = list(df.columns)
        from_dataframe(df, freq=Freq.hours)
        assert list(df.columns) == original_cols

    def test_exported_from_package(self):
        from torch_timeseries.dataset import DataFrameDataset as DFD, from_dataframe as fdf
        assert DFD is DataFrameDataset
        assert fdf is from_dataframe
