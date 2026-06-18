from __future__ import annotations
import numpy as np
from .base import IrregularTimeSeriesDataset


class UEAIrregular(IrregularTimeSeriesDataset):
    """Wraps any UEA dataset and applies synthetic timestamp dropout.

    Converts regular fixed-length sequences to irregular by randomly removing
    observations. All features at a dropped timestep are removed together.

    Args:
        dataset:   A UEA dataset with ``train_df``, ``test_df``,
                   ``train_labels``, ``test_labels`` attributes.
        drop_rate: Fraction of timesteps to drop (0.0 = no dropout).
        seed:      Random seed for reproducibility.
    """

    def __init__(self, dataset, drop_rate: float = 0.3, seed: int = 42) -> None:
        self._uea = dataset
        self.drop_rate = drop_rate
        self.seed = seed
        self._load()

    def download(self) -> None:
        pass

    def _load(self) -> None:
        rng = np.random.default_rng(self.seed)
        all_samples, all_times, all_masks, all_labels = [], [], [], []

        def _process_split(df, labels):
            ids = list(df.index.get_level_values("sample_id").unique())
            for sid in ids:
                x = df.loc[sid].values.astype(np.float32)   # (T, F)
                T, F = x.shape
                n_keep = max(1, int(T * (1.0 - self.drop_rate)))
                kept = np.sort(rng.choice(T, size=n_keep, replace=False))
                x_kept = x[kept]
                t_kept = kept.astype(np.float32)
                mask = np.ones_like(x_kept)

                all_samples.append(x_kept)
                all_times.append(t_kept)
                all_masks.append(mask)
                all_labels.append(int(labels.loc[sid]))

        _process_split(self._uea.train_df, self._uea.train_labels)
        _process_split(self._uea.test_df, self._uea.test_labels)

        self.samples = all_samples
        self.times = all_times
        self.masks = all_masks
        self.labels = np.array(all_labels, dtype=np.int64)
        self.num_features = self._uea.num_features
        self.num_classes = self._uea.num_classes
