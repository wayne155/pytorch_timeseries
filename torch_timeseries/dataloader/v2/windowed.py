"""Single Dataset that subsumes ``MultiStepTimeFeatureSet`` / ``MultivariateFast``
/ ``NoneOverlapWindowTS`` etc. behind one ``stride`` parameter.

* ``stride=1``        -> every sliding-window position (the v1 default).
* ``stride=W+H+S-1``  -> non-overlapping windows (the v1 ``MultivariateFast`` /
  ``NoneOverlapWindowTS`` behaviour).
* anything in between -> sub-sample sliding windows.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from torch_timeseries.core import TimeseriesSubset
from torch_timeseries.utils.timefeatures import time_features

from .batch import TSBatch


def _resolve_columns(col_spec, df) -> list:
    """Convert a list of int indices or string names to int indices.

    df must be the feature DataFrame (no 'date' column).
    Returns None if col_spec is None.
    """
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


class WindowedDataset(Dataset):
    """Produce ``TSBatch`` samples from a contiguous time-series subset.

    Parameters
    ----------
    subset : TimeseriesSubset
        A contiguous slice of the underlying dataset (already split).
    scaler : Scaler
        Already fitted scaler; we only ``transform`` here.
    window, horizon, steps : int
        Window length, horizon offset, and number of target steps.
    stride : int, default 1
        Step between successive windows.
    include_raw : bool
        Whether to populate ``x_raw`` / ``y_raw`` with un-scaled values.
    include_time : bool
        Whether to populate ``x_time`` / ``y_time`` with date-encoded features.
    include_index : bool
        Whether to populate ``x_index`` / ``y_index`` with integer time indices.
    single_variate : bool
        If True, samples are produced per feature channel and concatenated,
        matching the v1 ``single_variate`` semantics.
    time_enc, freq : passed through to ``time_features``.
    """

    def __init__(
        self,
        subset: TimeseriesSubset,
        scaler,
        window: int = 168,
        horizon: int = 1,
        steps: int = 1,
        stride: int = 1,
        include_raw: bool = True,
        include_time: bool = True,
        include_index: bool = False,
        single_variate: bool = False,
        time_enc: int = 0,
        freq: Optional[str] = None,
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
        self.include_raw = include_raw
        self.include_time = include_time
        self.include_index = include_index
        self.single_variate = single_variate
        self.num_features = subset.num_features
        self.freq = freq if freq is not None else subset.freq

        # Build feature DataFrame for name->index resolution (no 'date' col)
        _feat_df = subset.dataset.df.drop("date", axis=1, errors="ignore") if subset.dataset.df is not None else None
        self.input_columns  = _resolve_columns(input_columns,  _feat_df)
        self.target_columns = _resolve_columns(target_columns, _feat_df)

        if self.single_variate and (self.input_columns is not None or self.target_columns is not None):
            raise ValueError(
                "single_variate mode and input_columns/target_columns are mutually exclusive"
            )

        if self.input_columns is not None:
            self.num_features = len(self.input_columns)

        self._raw = subset.data
        self._scaled = scaler.transform(self._raw)

        if include_time:
            self._time = time_features(subset.dates, time_enc, self.freq)
        else:
            self._time = None

        if include_index:
            self._index = subset.time_index
        else:
            self._index = None

        n_starts = (len(subset) - total_len) // stride + 1
        self._n_starts = n_starts
        self._total_len = total_len

    def __len__(self) -> int:
        if self.single_variate:
            return self._n_starts * self.num_features
        return self._n_starts

    def _start_and_channel(self, index: int):
        if self.single_variate:
            channel = index // self._n_starts
            pos = index % self._n_starts
        else:
            channel = None
            pos = index
        return pos * self.stride, channel

    def __getitem__(self, index: int) -> TSBatch:
        if not isinstance(index, (int, np.integer)):
            raise TypeError(f"unsupported index type: {type(index)}")
        start, ch = self._start_and_channel(int(index))

        x_slice = slice(start, start + self.window)
        y_start = start + self.window + self.horizon - 1
        y_slice = slice(y_start, y_start + self.steps)

        x = self._scaled[x_slice]
        y = self._scaled[y_slice]
        if ch is not None:
            x = x[:, ch:ch + 1]
            y = y[:, ch:ch + 1]
        else:
            if self.input_columns is not None:
                x = x[:, self.input_columns]
            if self.target_columns is not None:
                y = y[:, self.target_columns]

        def _slice(arr, x_cols=None, y_cols=None, is_data=True):
            if arr is None:
                return None, None
            xv, yv = arr[x_slice], arr[y_slice]
            if is_data and ch is not None and arr.ndim == 2:
                xv, yv = xv[:, ch:ch + 1], yv[:, ch:ch + 1]
            else:
                if x_cols is not None and arr.ndim == 2:
                    xv = xv[:, x_cols]
                if y_cols is not None and arr.ndim == 2:
                    yv = yv[:, y_cols]
            return xv, yv

        x_raw, y_raw = _slice(self._raw, self.input_columns, self.target_columns, is_data=True)
        x_time, y_time = _slice(self._time, is_data=False)
        x_index, y_index = _slice(self._index, is_data=False)

        return TSBatch(
            x=x, y=y,
            x_raw=x_raw, y_raw=y_raw,
            x_time=x_time, y_time=y_time,
            x_index=x_index, y_index=y_index,
        )

    def transform(self, values):
        return self.scaler.transform(values)

    def inverse_transform(self, values):
        return self.scaler.inverse_transform(values)
