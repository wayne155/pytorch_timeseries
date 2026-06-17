"""Synthetic sinusoidal dataset for time series generation benchmarks.

Follows the TimeGAN convention: each feature is a sine wave with a randomly
drawn frequency, so the dataset tests whether a generative model can reproduce
the marginal distribution and autocorrelation structure of sinusoids.
"""
import os
import numpy as np
import pandas as pd

from ..core.dataset.dataset import TimeSeriesDataset, Freq


class Sine(TimeSeriesDataset):
    """Synthetic multi-variate sinusoidal dataset (TimeGAN benchmark).

    Generates a single long time series where each channel is a sine wave
    with a fixed random frequency and phase. Sliding-window dataloaders then
    slice it into independent windows for training and evaluation.

    Attributes:
        n_points: total time steps in the series (default 10 000).
        n_features: number of independent sine channels (default 5).
    """

    name: str = "Sine"
    freq: Freq = "h"

    def __init__(self, n_points: int = 10_000, n_features: int = 5, **kwargs):
        self.n_points = n_points
        self.n_features = n_features
        super().__init__(**kwargs)

    def download(self):
        pass   # generated synthetically — nothing to download

    def _load(self) -> np.ndarray:
        rng = np.random.default_rng(seed=42)
        t = np.linspace(0, 4 * np.pi, self.n_points)
        cols = {}
        for i in range(self.n_features):
            freq  = rng.uniform(1.0, 5.0)
            phase = rng.uniform(0.0, 2 * np.pi)
            cols[f"sine_{i}"] = np.sin(freq * t + phase)

        self.df    = pd.DataFrame(cols)
        self.df.insert(0, "date", pd.date_range("2020-01-01", periods=self.n_points, freq="h"))
        self.dates = pd.DataFrame({"date": self.df["date"]})
        self.data  = self.df.drop("date", axis=1).to_numpy().astype(np.float32)
        return self.data
