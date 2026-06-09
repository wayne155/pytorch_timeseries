"""Single Dataset that subsumes ``MultiStepTimeFeatureSet`` / ``MultivariateFast``
/ ``NoneOverlapWindowTS`` etc. behind one ``stride`` parameter.

* ``stride=1``        -> every sliding-window position (the v1 default).
* ``stride=W+H+S-1``  -> non-overlapping windows (the v1 ``MultivariateFast`` /
  ``NoneOverlapWindowTS`` behaviour).
* anything in between -> sub-sample sliding windows.
"""
from __future__ import annotations

from dataclasses import fields as dc_fields
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from torch_timeseries.core import TimeseriesSubset

from .batch import TSBatch, Time, TimeEncConfig


def _resolve_columns(col_spec, df) -> list:
    """Convert a list of int indices or string names to int indices."""
    if col_spec is None:
        return None
    if len(col_spec) == 0:
        raise ValueError("col_spec must not be empty; use None to include all columns")
    resolved = []
    cols = list(df.columns) if df is not None else []
    for c in col_spec:
        if isinstance(c, str):
            if c not in cols:
                raise ValueError(f"Column '{c}' not found. Available: {cols}")
            resolved.append(cols.index(c))
        else:
            idx = int(c)
            if cols and not (0 <= idx < len(cols)):
                raise ValueError(
                    f"Column index {idx} is out of range for dataset with {len(cols)} features"
                )
            resolved.append(idx)
    return resolved


def _slice_time(time: Optional[Time], sl) -> Optional[Time]:
    if time is None:
        return None
    return Time(**{f.name: getattr(time, f.name)[sl] for f in dc_fields(Time)})


class WindowedDataset(Dataset):
    """Produce ``TSBatch`` samples from a contiguous time-series subset.

    All modalities (raw, time, index) are always included in the batch.
    ``x_time_feature`` / ``y_time_feature`` (encoded float tensor) and
    ``x_time`` / ``y_time`` (structured ``Time``) are ``None`` only when
    the dataset has no date information.

    Parameters
    ----------
    subset : TimeseriesSubset
    scaler : Scaler
        Already fitted scaler; we only ``transform`` here.
    window, horizon, steps : int
    stride : int, default 1
    time_enc_cfg : TimeEncConfig, optional
        Encoding settings; defaults to ``TimeEncConfig()`` (calendar encoding).
    input_columns, target_columns : list of int or str, optional
    """

    def __init__(
        self,
        subset: TimeseriesSubset,
        scaler,
        window: int = 168,
        horizon: int = 1,
        steps: int = 1,
        stride: int = 1,
        time_enc_cfg: TimeEncConfig = None,
        input_columns=None,
        target_columns=None,
    ) -> None:
        if window <= 0 or horizon <= 0 or steps <= 0 or stride <= 0:
            raise ValueError("window/horizon/steps/stride must all be positive")
        total_len = window + horizon + steps - 1
        if len(subset) < total_len:
            raise ValueError(
                f"subset too short: need >= {total_len}, got {len(subset)}"
            )

        self.subset = subset
        self.scaler = scaler
        self.window = window
        self.horizon = horizon
        self.steps = steps
        self.stride = stride
        self.num_features = subset.num_features

        _feat_df = subset.dataset.df.drop("date", axis=1, errors="ignore") if subset.dataset.df is not None else None
        self.input_columns = _resolve_columns(input_columns, _feat_df)
        self.target_columns = _resolve_columns(target_columns, _feat_df)

        if self.input_columns is not None:
            self.num_features = len(self.input_columns)

        _cfg = time_enc_cfg or TimeEncConfig()
        self._raw = subset.data
        self._scaled = scaler.transform(self._raw)
        self._index = subset.time_index

        if subset.dates is not None:
            try:
                self._time_feature = _cfg.encode(subset.dates, subset.freq)
            except Exception:
                self._time_feature = None
            try:
                self._time = Time.from_dates(subset.dates)
            except Exception:
                self._time = None
        else:
            self._time_feature = None
            self._time = None

        n_starts = (len(subset) - total_len) // stride + 1
        self._n_starts = n_starts

    def __len__(self) -> int:
        return self._n_starts

    def __getitem__(self, index: int) -> TSBatch:
        if not isinstance(index, (int, np.integer)):
            raise TypeError(f"unsupported index type: {type(index)}")
        start = int(index) * self.stride

        x_slice = slice(start, start + self.window)
        y_start = start + self.window + self.horizon - 1
        y_slice = slice(y_start, y_start + self.steps)

        x = self._scaled[x_slice]
        y = self._scaled[y_slice]
        x_raw = self._raw[x_slice]
        y_raw = self._raw[y_slice]

        if self.input_columns is not None:
            x = x[:, self.input_columns]
            x_raw = x_raw[:, self.input_columns]
        if self.target_columns is not None:
            y = y[:, self.target_columns]
            y_raw = y_raw[:, self.target_columns]

        tf = self._time_feature
        return TSBatch(
            x=x, y=y,
            x_raw=x_raw, y_raw=y_raw,
            x_time_feature=tf[x_slice] if tf is not None else None,
            y_time_feature=tf[y_slice] if tf is not None else None,
            x_time=_slice_time(self._time, x_slice),
            y_time=_slice_time(self._time, y_slice),
            x_index=self._index[x_slice],
            y_index=self._index[y_slice],
        )

    def transform(self, values):
        return self.scaler.transform(values)

    def inverse_transform(self, values):
        return self.scaler.inverse_transform(values)
