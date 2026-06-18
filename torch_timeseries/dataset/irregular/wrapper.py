from __future__ import annotations
import numpy as np
from .base import IrregularTimeSeriesDataset


class IrregularWrapper(IrregularTimeSeriesDataset):
    """Wraps any ``TimeSeriesDataset`` and applies random timestamp dropout.

    Slices the dataset into non-overlapping windows of ``window`` timesteps,
    then drops ``drop_rate`` fraction of timesteps from each window to simulate
    irregular sampling. Used primarily with ``IrregularForecastDataModule``.

    Args:
        dataset:   Any ``TimeSeriesDataset`` with a ``.data`` numpy array.
        window:    Length of each non-overlapping window (default 96).
        drop_rate: Fraction of timesteps to drop per window.
        seed:      Random seed.
    """

    def __init__(self, dataset, window: int = 96,
                 drop_rate: float = 0.3, seed: int = 42) -> None:
        self._ts = dataset
        self.window = window
        self.drop_rate = drop_rate
        self.seed = seed
        self._load()

    def download(self) -> None:
        pass

    def _load(self) -> None:
        rng = np.random.default_rng(self.seed)
        data = self._ts.data                  # (N_total, F)
        N, F = data.shape
        n_windows = N // self.window

        all_samples, all_times, all_masks = [], [], []
        for i in range(n_windows):
            seg = data[i * self.window: (i + 1) * self.window].astype(np.float32)
            T = seg.shape[0]
            n_keep = max(1, int(T * (1.0 - self.drop_rate)))
            kept = np.sort(rng.choice(T, size=n_keep, replace=False))
            x_kept = seg[kept]
            t_kept = kept.astype(np.float32)
            mask = np.ones_like(x_kept)

            all_samples.append(x_kept)
            all_times.append(t_kept)
            all_masks.append(mask)

        self.samples = all_samples
        self.times = all_times
        self.masks = all_masks
        self.labels = None
        self.num_features = F
        self.num_classes = 0
