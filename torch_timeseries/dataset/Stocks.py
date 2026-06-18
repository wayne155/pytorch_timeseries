"""Google stock dataset for time series generation benchmarks (TimeGAN convention).

Downloads the stock_data.csv released alongside the TimeGAN paper
(Yoon et al., 2019 — https://github.com/jsyoon0823/TimeGAN).
Six features: Open, High, Low, Close, Adj Close, Volume — pre-normalised to [0, 1].
"""
import os
import numpy as np
import pandas as pd

from ..core.dataset.dataset import TimeSeriesDataset, Freq
from .utils import download_url


class Stocks(TimeSeriesDataset):
    """Google daily stock prices, 2004-2019 (TimeGAN generation benchmark).

    Six features: Open, High, Low, Close, Adj Close, Volume — all normalised
    to [0, 1] as in the original TimeGAN paper.
    """

    name: str = "Stocks"
    freq: Freq = "d"

    _URL = (
        "https://raw.githubusercontent.com/jsyoon0823/TimeGAN"
        "/master/data/stock_data.csv"
    )
    _FILENAME = "stock_data.csv"

    def download(self):
        download_url(self._URL, self.dir, filename=self._FILENAME)

    def _load(self) -> np.ndarray:
        self.file_path = os.path.join(self.dir, self._FILENAME)
        raw = pd.read_csv(self.file_path)
        feature_cols = [c for c in raw.columns if raw[c].dtype.kind in ("f", "i")]
        values = raw[feature_cols].to_numpy().astype(np.float64)
        # Per-feature min-max normalisation to [0, 1] (TimeGAN convention).
        col_min = values.min(axis=0, keepdims=True)
        col_max = values.max(axis=0, keepdims=True)
        values = (values - col_min) / (col_max - col_min + 1e-8)
        n = len(values)
        self.df = pd.DataFrame(values, columns=feature_cols)
        self.df.insert(0, "date", pd.date_range("2004-08-19", periods=n, freq="B"))
        self.dates = pd.DataFrame({"date": self.df["date"]})
        self.data  = values.astype(np.float32)
        return self.data
