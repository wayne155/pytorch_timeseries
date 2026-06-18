from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .base import IrregularTimeSeriesDataset


class MIMIC(IrregularTimeSeriesDataset):
    """MIMIC-III / MIMIC-IV irregular time-series dataset.

    Cannot auto-download — requires credentialed PhysioNet access.
    Expects a pre-processed CSV at ``{data_dir}/mimic_processed.csv`` with columns::

        subject_id, hours_from_admit, <feature_cols...>, label

    Raises ``FileNotFoundError`` with instructions if ``data_dir`` is missing.
    """

    def __init__(self, data_dir: str, version: str = "III") -> None:
        self.version = version
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(
                f"MIMIC data directory not found: {data_dir}\n"
                "MIMIC requires credentialed access. Steps:\n"
                "  1. Register at https://physionet.org\n"
                "  2. Complete CITI training and request access to MIMIC-III or MIMIC-IV\n"
                "  3. Download and pre-process into mimic_processed.csv with columns:\n"
                "     subject_id, hours_from_admit, <features...>, label"
            )
        self.root = str(data_path.parent)
        self._data_dir = data_path
        self._load()

    def download(self) -> None:
        pass  # Cannot auto-download

    def _load(self) -> None:
        csv_path = self._data_dir / "mimic_processed.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected pre-processed file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        required = {"subject_id", "hours_from_admit", "label"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"mimic_processed.csv must have columns: {required}. Got: {list(df.columns)}"
            )

        feature_cols = [c for c in df.columns
                        if c not in ("subject_id", "hours_from_admit", "label")]
        F = len(feature_cols)

        all_samples, all_times, all_masks, all_labels = [], [], [], []
        for _, grp in df.groupby("subject_id"):
            grp = grp.sort_values("hours_from_admit")
            t_arr = grp["hours_from_admit"].values.astype(np.float32)
            x = grp[feature_cols].values.astype(np.float32)
            mask = (~pd.isna(grp[feature_cols])).values.astype(np.float32)
            x = np.nan_to_num(x, nan=0.0)
            label = int(grp["label"].iloc[-1])

            all_samples.append(x)
            all_times.append(t_arr)
            all_masks.append(mask)
            all_labels.append(label)

        self.samples = all_samples
        self.times = all_times
        self.masks = all_masks
        self.labels = np.array(all_labels, dtype=np.int64)
        self.num_features = F
        self.num_classes = len(np.unique(self.labels))
