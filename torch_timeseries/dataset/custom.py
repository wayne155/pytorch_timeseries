"""Build a TimeSeriesDataset directly from a local CSV file.

    from torch_timeseries.dataset import build_dataset
    dataset = build_dataset(csv="./my_sensors.csv", freq="h")

The CSV needs a ``date`` column plus one column per variable; everything else
(num_features, length, time index) is inferred from the file.
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
