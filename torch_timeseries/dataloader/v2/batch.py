"""Structured batch type for forecasting / windowed time-series tasks."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

ArrayLike = "np.ndarray | Tensor"


@dataclass
class TimeEncConfig:
    """Time encoding configuration.

    Either supply ``custom_fn`` for a bespoke encoder, or set ``time_enc``
    and ``freq`` to use the standard ``time_features`` encoding.

    ``custom_fn`` signature: ``(dates: pd.DataFrame) -> np.ndarray``
    """
    time_enc: Union[str, int] = "calendar"
    freq: Optional[str] = None
    custom_fn: Optional[Callable] = field(default=None, repr=False)

    def encode(self, dates, dataset_freq: Optional[str] = None):
        """Return an ``[T, C]`` float32 array, or ``None`` if *dates* is ``None``."""
        if dates is None:
            return None
        if self.custom_fn is not None:
            return self.custom_fn(dates)
        from torch_timeseries.utils.timefeatures import time_features
        freq = self.freq if self.freq is not None else dataset_freq
        return time_features(dates, self.time_enc, freq).astype("float32")


@dataclass
class Time:
    """Named time components for a window of timesteps.

    Each field is a LongTensor of shape [T] (per-item) or [B, T] (batched).
    Access: batch.x_time.year, batch.x_time.month, etc.
    None if the component is not available for that dataset.
    """
    year: Optional[Tensor] = None
    month: Optional[Tensor] = None
    day: Optional[Tensor] = None
    weekday: Optional[Tensor] = None
    hour: Optional[Tensor] = None
    minute: Optional[Tensor] = None
    second: Optional[Tensor] = None

    def to(self, device, non_blocking: bool = False) -> "Time":
        return Time(**{
            f.name: getattr(self, f.name).to(device=device, non_blocking=non_blocking)
            if getattr(self, f.name) is not None else None
            for f in fields(self)
        })

    @staticmethod
    def from_dates(dates) -> "Time":
        """Build from a pandas DataFrame with a 'date' column."""
        import pandas as pd
        if hasattr(dates, "columns") and "date" in dates.columns:
            dt = pd.DatetimeIndex(dates["date"].values)
        else:
            dt = pd.DatetimeIndex(dates)
        to_t = lambda arr: torch.tensor(arr, dtype=torch.long)
        return Time(
            year=to_t(dt.year.to_numpy()),
            month=to_t(dt.month.to_numpy()),
            day=to_t(dt.day.to_numpy()),
            weekday=to_t(dt.dayofweek.to_numpy()),
            hour=to_t(dt.hour.to_numpy()),
            minute=to_t(dt.minute.to_numpy()),
            second=to_t(dt.second.to_numpy()),
        )


@dataclass
class TSBatch:
    x: Optional[Tensor] = None
    y: Optional[Tensor] = None
    x_raw: Optional[Tensor] = None
    y_raw: Optional[Tensor] = None
    x_time_feature: Optional[Tensor] = None
    y_time_feature: Optional[Tensor] = None
    x_time: Optional[Time] = None
    y_time: Optional[Time] = None
    x_index: Optional[Tensor] = None
    y_index: Optional[Tensor] = None

    def as_tuple(self, keys: Sequence[str]) -> Tuple:
        """Return selected fields as a positional tuple (for legacy unpacking)."""
        return tuple(getattr(self, k) for k in keys)

    def to(self, device, non_blocking: bool = False) -> "TSBatch":
        kwargs = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, Tensor):
                v = v.to(device=device, non_blocking=non_blocking)
            elif isinstance(v, Time):
                v = v.to(device, non_blocking)
            kwargs[f.name] = v
        return TSBatch(**kwargs)

    def keys(self) -> List[str]:
        return [f.name for f in fields(self) if getattr(self, f.name) is not None]


def _stack(values: Iterable):
    values = list(values)
    if values[0] is None:
        return None
    if isinstance(values[0], Tensor):
        return torch.stack(values, dim=0)
    return torch.from_numpy(np.stack(values, axis=0))


def _stack_time(times: Sequence) -> Optional[Time]:
    non_none = [t for t in times if t is not None]
    if not non_none:
        return None
    kwargs = {}
    for f in fields(Time):
        vals = [getattr(t, f.name) for t in non_none]
        kwargs[f.name] = torch.stack(vals, dim=0) if vals[0] is not None else None
    return Time(**kwargs)


def collate_tsbatch(samples: Sequence[TSBatch]) -> TSBatch:
    """Default collate fn for ``DataLoader``."""
    out = {}
    for f in fields(TSBatch):
        vals = [getattr(s, f.name) for s in samples]
        if any(isinstance(v, Time) for v in vals):
            out[f.name] = _stack_time(vals)
        else:
            out[f.name] = _stack(vals)
    return TSBatch(**out)
