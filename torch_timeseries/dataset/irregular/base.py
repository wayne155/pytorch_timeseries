from __future__ import annotations
from typing import List, Optional
import numpy as np


class IrregularTimeSeriesDataset:
    """Base class for irregular time-series datasets.

    Subclasses implement ``download()`` and ``_load()``.
    After ``_load()``, the following attributes must be set:

        samples:      List[np.ndarray]  — [(T_i, F)]  variable-length per sample
        times:        List[np.ndarray]  — [(T_i,)]    raw observation times (any unit)
        masks:        List[np.ndarray]  — [(T_i, F)]  1=observed, 0=missing
        labels:       Optional[np.ndarray]  — (N,) integer class labels; None if no labels
        num_features: int
        num_classes:  int               — 0 if no labels
    """

    samples: List[np.ndarray]
    times: List[np.ndarray]
    masks: List[np.ndarray]
    labels: Optional[np.ndarray]
    num_features: int
    num_classes: int

    def __init__(self, root: str, download: bool = True) -> None:
        self.root = root
        if download:
            self.download()
        self._load()

    def download(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} must implement download()")

    def _load(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} must implement _load()")

    def __len__(self) -> int:
        return len(self.samples)
