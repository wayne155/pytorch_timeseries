"""Build a TimeSeriesDataset from a local CSV file or an in-memory DataFrame.

    # From a CSV file
    from torch_timeseries.dataset import build_dataset
    dataset = build_dataset(csv="./my_sensors.csv", freq="h")

    # From an in-memory DataFrame
    from torch_timeseries.dataset import from_dataframe
    import pandas as pd
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=100, freq="h"),
                        "feat1": ..., "feat2": ...})
    dataset = from_dataframe(df, freq="h")

Both helpers accept ``date_column`` and ``name`` keyword arguments. Everything
else (num_features, length, time index) is inferred automatically.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from ..core.dataset.dataset import Freq, TimeSeriesDataset


class CSVDataset(TimeSeriesDataset):
    """A TimeSeriesDataset backed by a local CSV file (no download step)."""

    def __init__(
        self,
        csv: str,
        freq: Freq,
        name: Optional[str] = None,
        date_column: str = "date",
    ) -> None:
        csv = os.path.expanduser(csv)
        if not os.path.isfile(csv):
            raise FileNotFoundError(f"CSV file not found: {csv}")
        self.file_path = csv
        self.freq = freq
        self.name = name or os.path.splitext(os.path.basename(csv))[0]
        self.date_column = date_column
        # The file already exists locally; bypass the root/download machinery.
        self.root = os.path.dirname(csv)
        self.dir = self.root
        self.columns = []

        self._load()
        self.num_features = self.data.shape[1]
        self.length = self.data.shape[0]
        self.time_index = np.arange(len(self.df))

    def download(self):
        pass

    def _load(self) -> np.ndarray:
        df = pd.read_csv(self.file_path, parse_dates=[self.date_column])
        if self.date_column != "date":
            df = df.rename(columns={self.date_column: "date"})
        self.df = df
        self.dates = pd.DataFrame({"date": df["date"]})
        self.data = df.drop("date", axis=1).to_numpy()
        return self.data


def build_dataset(
    csv: str,
    freq: Freq,
    name: Optional[str] = None,
    date_column: str = "date",
) -> CSVDataset:
    """Create a dataset from a CSV file with a date column.

    Args:
        csv: path to the CSV file.
        freq: pandas-style frequency of the rows ('h', 't', 'd', ...).
        name: dataset name; defaults to the CSV file name.
        date_column: name of the timestamp column (renamed to 'date').
    """
    return CSVDataset(csv=csv, freq=freq, name=name, date_column=date_column)


class DataFrameDataset(TimeSeriesDataset):
    """A TimeSeriesDataset backed by an in-memory pandas DataFrame.

    Args:
        df: DataFrame with a timestamp column and one column per feature.
        freq: pandas-style frequency string (``'h'``, ``'t'``, ``'d'``, …).
        name: human-readable dataset name (default ``'dataframe'``).
        date_column: name of the timestamp column.  If the column doesn't exist
            and the DataFrame has a :class:`~pandas.DatetimeIndex`, that is used
            instead.
    """

    def __init__(
        self,
        df: "pd.DataFrame",
        freq: Freq,
        name: Optional[str] = None,
        date_column: str = "date",
    ) -> None:
        self._source_df = df.copy()
        self.freq = freq
        self.name = name or "dataframe"
        self.date_column = date_column
        # Bypass the root/download machinery — data is already in memory.
        self.root = ""
        self.dir = ""
        self.columns = []

        self._load()
        self.num_features = self.data.shape[1]
        self.length = self.data.shape[0]
        self.time_index = np.arange(self.length)

    def download(self):
        pass

    def _load(self) -> np.ndarray:
        df = self._source_df.copy()
        if self.date_column in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
                df[self.date_column] = pd.to_datetime(df[self.date_column])
            if self.date_column != "date":
                df = df.rename(columns={self.date_column: "date"})
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": "date"})
        else:
            raise ValueError(
                f"DataFrame has no column '{self.date_column}' and no DatetimeIndex. "
                "Provide a date column or pass a DataFrame with a DatetimeIndex."
            )
        self.df = df
        self.dates = pd.DataFrame({"date": df["date"]})
        self.data = df.drop("date", axis=1).to_numpy().astype(np.float32)
        return self.data


def from_dataframe(
    df: "pd.DataFrame",
    freq: Freq,
    name: Optional[str] = None,
    date_column: str = "date",
) -> DataFrameDataset:
    """Create a dataset from an in-memory pandas DataFrame.

    Args:
        df: DataFrame with a timestamp column (or DatetimeIndex) and one column
            per feature.
        freq: pandas-style frequency of the rows (``'h'``, ``'t'``, ``'d'``, …).
        name: dataset name; defaults to ``'dataframe'``.
        date_column: name of the timestamp column.  Falls back to a
            :class:`~pandas.DatetimeIndex` if the column is missing.
    """
    return DataFrameDataset(df=df, freq=freq, name=name, date_column=date_column)
