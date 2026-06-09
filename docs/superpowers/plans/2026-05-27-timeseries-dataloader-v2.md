# Time Series Dataloader V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the forecasting/imputation time-series data path around explicit feature selection, structured batch dictionaries, train-only scaling, reusable split strategies, and decoupled time feature dimensions while keeping old experiments working during migration.

**Architecture:** Introduce a v2 data layer under `torch_timeseries/dataloader/v2/` with small units: feature selection, batch collate helpers, split strategies, scaler bundles, time feature encoders, window specs, windowed datasets, and a data module that orchestrates them. Keep existing `SlidingWindowTS`, `ETTHLoader`, `ETTMLoader`, and experiment tuple consumers intact initially by adding compatibility adapters before migrating experiments to the new dict batch contract.

**Tech Stack:** Python dataclasses, NumPy, pandas, PyTorch `Dataset`/`DataLoader`, existing `torch_timeseries.scaler` classes, pytest.

---

## Design Summary

### Current Problems

The current data path mixes too many responsibilities:

- `SlidingWindowTS` owns split policy, scaler fitting, wrapper selection, time encoding, raw output options, and `DataLoader` creation.
- `ETTHLoader` and `ETTMLoader` subclass `SlidingWindowTS` only to override fixed split boundaries.
- `include_raw`, `time_index`, and `fast_*` alter tuple length and semantics, while experiments hard-code tuple unpacking.
- `freq` is used both as dataset sampling metadata and as the key for time feature dimensionality in `timefeatures.py` and `embedding.py`.
- `single_variate` means channel-independent sampling, not multi-variate-to-single-target forecasting, but those concepts are easy to confuse.
- Custom datasets must inherit `TimeSeriesDataset` and populate several implicit fields.

### New Concepts

- `FeatureSelector = int | str | Sequence[int | str] | None`
  - `None` means all features.
  - `int` supports negative indexing.
  - `str` resolves against feature names.
  - Datasets without names receive generated names: `feature_0`, `feature_1`, ...
- `input_features` and `target_features` are independent.
  - M->1: `input_features=None`, `target_features=[-1]`.
  - M->M: `input_features=None`, `target_features=None`.
  - M->K: `input_features=None`, `target_features=[0, 3]` or names.
- `Dataset.__getitem__` returns a PyTorch-native `dict[str, Tensor]`, not a positional tuple.
- Optional fields are omitted from the dict. Training code uses `batch.get("raw_y")`.
- `TimeSeriesBatch` can exist as a convenience wrapper at the collate or experiment layer, but default v2 behavior should remain dict-based to match PyTorch default collation.
- Raw values are standardized fields (`raw_x`, `raw_y`) instead of changing tuple arity.
- Scalers are fit once on train split only, managed by `ScalerBundle`.
- Time feature encoders expose `output_dim`; `TimeFeatureEmbedding` should accept `input_dim` for `timeF` instead of inferring from `freq`.

---

## Target File Structure

- Create `torch_timeseries/dataloader/v2/__init__.py`
  - Public exports for v2 data utilities.
- Create `torch_timeseries/dataloader/v2/typing.py`
  - Type aliases and dataclasses shared by v2 modules.
- Create `torch_timeseries/dataloader/v2/features.py`
  - Feature name normalization and `FeatureSelector` resolution.
- Create `torch_timeseries/dataloader/v2/splits.py`
  - `SplitIndices`, `RatioSplit`, `FixedBoundarySplit`.
- Create `torch_timeseries/dataloader/v2/scalers.py`
  - `ScalerBundle` for independent input/target scaling.
- Create `torch_timeseries/dataloader/v2/time_features.py`
  - Non-mutating `TimeFeatureEncoder` with `output_dim`.
- Create `torch_timeseries/dataloader/v2/window.py`
  - `WindowSpec` and `WindowedDataset`.
- Create `torch_timeseries/dataloader/v2/module.py`
  - `TimeSeriesDataModule` that returns train/val/test `DataLoader`s.
- Create `torch_timeseries/dataloader/v2/adapters.py`
  - Old tuple compatibility adapter for legacy experiments.
- Modify `torch_timeseries/dataloader/__init__.py`
  - Export v2 utilities without removing old exports.
- Modify `torch_timeseries/utils/timefeatures.py`
  - Fix deprecated/non-mutating behavior after v2 tests cover it.
- Modify `torch_timeseries/nn/embedding.py`
  - Add `input_dim` support to `TimeFeatureEmbedding` and `DataEmbedding`.
- Later modify `torch_timeseries/dataloader/sliding_window_ts.py` and `torch_timeseries/dataloader/ETT.py`
  - Reimplement through v2 compatibility adapter.
- Later modify `torch_timeseries/experiments/forecast.py` and `torch_timeseries/experiments/imputation.py`
  - Consume dict batches directly.

---

## Task 1: Feature Selection Contract

**Files:**
- Create: `torch_timeseries/dataloader/v2/__init__.py`
- Create: `torch_timeseries/dataloader/v2/typing.py`
- Create: `torch_timeseries/dataloader/v2/features.py`
- Test: `tests/dataloader/v2/test_features.py`

- [ ] **Step 1: Write failing tests for selector normalization**

Create `tests/dataloader/v2/test_features.py`:

```python
import pytest

from torch_timeseries.dataloader.v2.features import FeatureSelection, resolve_feature_selection


def test_none_selector_defaults_to_all_generated_names():
    selection = resolve_feature_selection(
        selector=None,
        num_features=3,
        feature_names=None,
        role="input_features",
    )

    assert selection == FeatureSelection(
        indices=[0, 1, 2],
        names=["feature_0", "feature_1", "feature_2"],
    )


def test_selector_supports_negative_indices():
    selection = resolve_feature_selection(
        selector=[0, -1],
        num_features=4,
        feature_names=None,
        role="target_features",
    )

    assert selection.indices == [0, 3]
    assert selection.names == ["feature_0", "feature_3"]


def test_selector_supports_names():
    selection = resolve_feature_selection(
        selector=["load", "temp"],
        num_features=3,
        feature_names=["load", "holiday", "temp"],
        role="target_features",
    )

    assert selection.indices == [0, 2]
    assert selection.names == ["load", "temp"]


def test_single_int_selector_becomes_one_feature():
    selection = resolve_feature_selection(
        selector=-1,
        num_features=3,
        feature_names=None,
        role="target_features",
    )

    assert selection.indices == [2]
    assert selection.names == ["feature_2"]


def test_single_name_selector_becomes_one_feature():
    selection = resolve_feature_selection(
        selector="OT",
        num_features=2,
        feature_names=["HUFL", "OT"],
        role="target_features",
    )

    assert selection.indices == [1]
    assert selection.names == ["OT"]


def test_duplicate_feature_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate feature name"):
        resolve_feature_selection(
            selector=None,
            num_features=2,
            feature_names=["OT", "OT"],
            role="input_features",
        )


def test_out_of_range_index_mentions_role():
    with pytest.raises(IndexError, match="target_features"):
        resolve_feature_selection(
            selector=[3],
            num_features=3,
            feature_names=None,
            role="target_features",
        )


def test_unknown_name_mentions_role_and_name():
    with pytest.raises(KeyError, match="target_features.*missing"):
        resolve_feature_selection(
            selector=["missing"],
            num_features=2,
            feature_names=["a", "b"],
            role="target_features",
        )
```

- [ ] **Step 2: Run the feature tests and verify they fail**

Run:

```bash
python -m pytest tests/dataloader/v2/test_features.py -q
```

Expected: FAIL because `torch_timeseries.dataloader.v2.features` does not exist.

- [ ] **Step 3: Add shared v2 typing**

Create `torch_timeseries/dataloader/v2/typing.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeAlias

FeatureKey: TypeAlias = int | str
FeatureSelector: TypeAlias = FeatureKey | Sequence[FeatureKey] | None


@dataclass(frozen=True)
class FeatureSelection:
    indices: list[int]
    names: list[str]
```

- [ ] **Step 4: Implement feature selector resolution**

Create `torch_timeseries/dataloader/v2/features.py`:

```python
from __future__ import annotations

from collections.abc import Sequence

from .typing import FeatureKey, FeatureSelection, FeatureSelector


def normalize_feature_names(
    num_features: int,
    feature_names: Sequence[str] | None = None,
) -> list[str]:
    if num_features <= 0:
        raise ValueError("num_features must be positive")

    if feature_names is None:
        return [f"feature_{idx}" for idx in range(num_features)]

    names = list(feature_names)
    if len(names) != num_features:
        raise ValueError(
            f"feature_names length {len(names)} does not match num_features {num_features}"
        )

    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"duplicate feature name: {name}")
        seen.add(name)
    return names


def resolve_feature_selection(
    selector: FeatureSelector,
    num_features: int,
    feature_names: Sequence[str] | None = None,
    role: str = "features",
) -> FeatureSelection:
    names = normalize_feature_names(num_features, feature_names)

    if selector is None:
        indices = list(range(num_features))
    elif isinstance(selector, (int, str)):
        indices = [_resolve_one(selector, names, role)]
    elif isinstance(selector, Sequence):
        indices = [_resolve_one(item, names, role) for item in selector]
    else:
        raise TypeError(
            f"{role} must be None, an int, a str, or a sequence of ints/strs"
        )

    selected_names = [names[idx] for idx in indices]
    return FeatureSelection(indices=indices, names=selected_names)


def _resolve_one(item: FeatureKey, names: list[str], role: str) -> int:
    if isinstance(item, int):
        index = item if item >= 0 else len(names) + item
        if index < 0 or index >= len(names):
            raise IndexError(
                f"{role} index {item} is out of range for {len(names)} features"
            )
        return index

    if isinstance(item, str):
        try:
            return names.index(item)
        except ValueError as exc:
            raise KeyError(f"{role} contains unknown feature name: {item}") from exc

    raise TypeError(f"{role} items must be int or str, got {type(item).__name__}")
```

- [ ] **Step 5: Export v2 symbols**

Create `torch_timeseries/dataloader/v2/__init__.py`:

```python
from .features import normalize_feature_names, resolve_feature_selection
from .typing import FeatureSelection, FeatureSelector

__all__ = [
    "FeatureSelection",
    "FeatureSelector",
    "normalize_feature_names",
    "resolve_feature_selection",
]
```

- [ ] **Step 6: Run feature tests**

Run:

```bash
python -m pytest tests/dataloader/v2/test_features.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add torch_timeseries/dataloader/v2 tests/dataloader/v2/test_features.py
git commit -m "feat: add v2 feature selection"
```

---

## Task 2: Split Strategies

**Files:**
- Create: `torch_timeseries/dataloader/v2/splits.py`
- Modify: `torch_timeseries/dataloader/v2/__init__.py`
- Test: `tests/dataloader/v2/test_splits.py`

- [ ] **Step 1: Write failing tests for ratio and fixed-boundary splits**

Create `tests/dataloader/v2/test_splits.py`:

```python
import pytest

from torch_timeseries.dataloader.v2.splits import (
    FixedBoundarySplit,
    RatioSplit,
    SplitIndices,
)


def test_ratio_split_without_overlap():
    split = RatioSplit(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    indices = split.split(length=10, context_length=3, horizon=1, prediction_length=2)

    assert indices == SplitIndices(
        train=list(range(0, 6)),
        val=list(range(6, 8)),
        test=list(range(8, 10)),
    )


def test_ratio_split_with_uniform_eval_overlap():
    split = RatioSplit(
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        uniform_eval=True,
    )
    indices = split.split(length=10, context_length=3, horizon=1, prediction_length=2)

    assert indices.train == list(range(0, 6))
    assert indices.val == list(range(1, 8))
    assert indices.test == list(range(3, 10))


def test_ratio_split_rejects_invalid_sum():
    with pytest.raises(ValueError, match="sum to 1.0"):
        RatioSplit(train_ratio=0.6, val_ratio=0.3, test_ratio=0.3)


def test_fixed_boundary_split_matches_ett_style_boundaries():
    split = FixedBoundarySplit(
        train=(0, 12),
        val=(8, 16),
        test=(12, 20),
    )
    indices = split.split(length=20, context_length=4, horizon=1, prediction_length=1)

    assert indices.train == list(range(0, 12))
    assert indices.val == list(range(8, 16))
    assert indices.test == list(range(12, 20))


def test_fixed_boundary_rejects_out_of_range():
    split = FixedBoundarySplit(
        train=(0, 12),
        val=(8, 16),
        test=(12, 21),
    )

    with pytest.raises(ValueError, match="test"):
        split.split(length=20, context_length=4, horizon=1, prediction_length=1)
```

- [ ] **Step 2: Run split tests and verify they fail**

Run:

```bash
python -m pytest tests/dataloader/v2/test_splits.py -q
```

Expected: FAIL because `splits.py` does not exist.

- [ ] **Step 3: Implement split strategies**

Create `torch_timeseries/dataloader/v2/splits.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Protocol


@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    val: list[int]
    test: list[int]


class SplitStrategy(Protocol):
    def split(
        self,
        length: int,
        context_length: int,
        horizon: int,
        prediction_length: int,
    ) -> SplitIndices:
        ...


@dataclass(frozen=True)
class RatioSplit:
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    uniform_eval: bool = False

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not isclose(total, 1.0, abs_tol=1e-8):
            raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0")
        if min(self.train_ratio, self.val_ratio, self.test_ratio) < 0:
            raise ValueError("split ratios must be non-negative")

    def split(
        self,
        length: int,
        context_length: int,
        horizon: int,
        prediction_length: int,
    ) -> SplitIndices:
        train_size = int(self.train_ratio * length)
        test_size = int(self.test_ratio * length)
        val_size = length - train_size - test_size

        train_start, train_end = 0, train_size
        val_start, val_end = train_end, train_end + val_size
        test_start, test_end = length - test_size, length

        if self.uniform_eval:
            overlap = context_length + horizon - 1
            val_start = max(0, val_start - overlap)
            test_start = max(0, test_start - overlap)

        return SplitIndices(
            train=list(range(train_start, train_end)),
            val=list(range(val_start, val_end)),
            test=list(range(test_start, test_end)),
        )


@dataclass(frozen=True)
class FixedBoundarySplit:
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]

    def split(
        self,
        length: int,
        context_length: int,
        horizon: int,
        prediction_length: int,
    ) -> SplitIndices:
        return SplitIndices(
            train=self._range("train", self.train, length),
            val=self._range("val", self.val, length),
            test=self._range("test", self.test, length),
        )

    @staticmethod
    def _range(name: str, boundary: tuple[int, int], length: int) -> list[int]:
        start, end = boundary
        if start < 0 or end < start or end > length:
            raise ValueError(
                f"{name} boundary {boundary} is invalid for dataset length {length}"
            )
        return list(range(start, end))
```

- [ ] **Step 4: Export split symbols**

Modify `torch_timeseries/dataloader/v2/__init__.py`:

```python
from .features import normalize_feature_names, resolve_feature_selection
from .splits import FixedBoundarySplit, RatioSplit, SplitIndices, SplitStrategy
from .typing import FeatureSelection, FeatureSelector

__all__ = [
    "FeatureSelection",
    "FeatureSelector",
    "FixedBoundarySplit",
    "RatioSplit",
    "SplitIndices",
    "SplitStrategy",
    "normalize_feature_names",
    "resolve_feature_selection",
]
```

- [ ] **Step 5: Run split and feature tests**

Run:

```bash
python -m pytest tests/dataloader/v2/test_features.py tests/dataloader/v2/test_splits.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add torch_timeseries/dataloader/v2 tests/dataloader/v2/test_splits.py
git commit -m "feat: add v2 split strategies"
```

---

## Task 3: ScalerBundle With Independent Input And Target Scaling

**Files:**
- Create: `torch_timeseries/dataloader/v2/scalers.py`
- Modify: `torch_timeseries/dataloader/v2/__init__.py`
- Test: `tests/dataloader/v2/test_scalers.py`

- [ ] **Step 1: Write failing scaler bundle tests**

Create `tests/dataloader/v2/test_scalers.py`:

```python
import numpy as np
import torch

from torch_timeseries.dataloader.v2.scalers import ScalerBundle
from torch_timeseries.scaler import StandardScaler


def test_scaler_bundle_fits_input_and_target_columns_independently():
    values = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [1000.0, 1000.0, 1000.0],
        ],
        dtype=np.float32,
    )

    bundle = ScalerBundle(StandardScaler)
    bundle.fit(values[:3], input_indices=[0, 1, 2], target_indices=[2])

    x = bundle.transform_input(values[:1, [0, 1, 2]])
    y = bundle.transform_target(values[:1, [2]])

    np.testing.assert_allclose(x, [[-1.2247449, -1.2247449, -1.2247449]], rtol=1e-6)
    np.testing.assert_allclose(y, [[-1.2247449]], rtol=1e-6)


def test_scaler_bundle_inverse_target_handles_m_to_one_torch_tensor():
    values = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
        ],
        dtype=np.float32,
    )
    bundle = ScalerBundle(StandardScaler)
    bundle.fit(values, input_indices=[0, 1, 2], target_indices=[2])

    scaled = torch.zeros(2, 4, 1)
    restored = bundle.inverse_transform_target(scaled)

    assert restored.shape == (2, 4, 1)
    assert torch.allclose(restored, torch.full((2, 4, 1), 200.0))
```

- [ ] **Step 2: Run scaler tests and verify they fail**

Run:

```bash
python -m pytest tests/dataloader/v2/test_scalers.py -q
```

Expected: FAIL because `scalers.py` does not exist.

- [ ] **Step 3: Implement ScalerBundle**

Create `torch_timeseries/dataloader/v2/scalers.py`:

```python
from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy

import numpy as np

from torch_timeseries.core.scaler import Scaler
from torch_timeseries.scaler import StandardScaler


class ScalerBundle:
    def __init__(
        self,
        scaler_factory: Callable[[], Scaler] | type[Scaler] | Scaler = StandardScaler,
    ) -> None:
        self.input_scaler = self._new_scaler(scaler_factory)
        self.target_scaler = self._new_scaler(scaler_factory)
        self.input_indices: list[int] | None = None
        self.target_indices: list[int] | None = None

    def fit(
        self,
        train_values: np.ndarray,
        input_indices: Sequence[int],
        target_indices: Sequence[int],
    ) -> None:
        self.input_indices = list(input_indices)
        self.target_indices = list(target_indices)
        self.input_scaler.fit(train_values[:, self.input_indices])
        self.target_scaler.fit(train_values[:, self.target_indices])

    def transform_input(self, values):
        return self.input_scaler.transform(values)

    def transform_target(self, values):
        return self.target_scaler.transform(values)

    def inverse_transform_input(self, values):
        return self.input_scaler.inverse_transform(values)

    def inverse_transform_target(self, values):
        return self.target_scaler.inverse_transform(values)

    @staticmethod
    def _new_scaler(scaler_factory):
        if isinstance(scaler_factory, Scaler):
            return deepcopy(scaler_factory)
        return scaler_factory()
```

- [ ] **Step 4: Export ScalerBundle**

Modify `torch_timeseries/dataloader/v2/__init__.py`:

```python
from .features import normalize_feature_names, resolve_feature_selection
from .scalers import ScalerBundle
from .splits import FixedBoundarySplit, RatioSplit, SplitIndices, SplitStrategy
from .typing import FeatureSelection, FeatureSelector

__all__ = [
    "FeatureSelection",
    "FeatureSelector",
    "FixedBoundarySplit",
    "RatioSplit",
    "ScalerBundle",
    "SplitIndices",
    "SplitStrategy",
    "normalize_feature_names",
    "resolve_feature_selection",
]
```

- [ ] **Step 5: Run scaler tests**

Run:

```bash
python -m pytest tests/dataloader/v2/test_scalers.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add torch_timeseries/dataloader/v2 tests/dataloader/v2/test_scalers.py
git commit -m "feat: add v2 scaler bundle"
```

---

## Task 4: Non-Mutating TimeFeatureEncoder

**Files:**
- Create: `torch_timeseries/dataloader/v2/time_features.py`
- Modify: `torch_timeseries/dataloader/v2/__init__.py`
- Test: `tests/dataloader/v2/test_time_features.py`

- [ ] **Step 1: Write failing tests for time feature encoding**

Create `tests/dataloader/v2/test_time_features.py`:

```python
import numpy as np
import pandas as pd

from torch_timeseries.dataloader.v2.time_features import TimeFeatureEncoder


def test_continuous_hourly_encoder_has_four_features_and_does_not_mutate_input():
    dates = pd.DataFrame(
        {"date": pd.date_range("2021-01-01", periods=3, freq="h")}
    )
    before_columns = list(dates.columns)

    encoder = TimeFeatureEncoder(mode="continuous", freq="h")
    encoded = encoder.fit_transform(dates)

    assert encoded.shape == (3, 4)
    assert encoder.output_dim == 4
    assert list(dates.columns) == before_columns


def test_calendar_minutely_encoder_supports_generated_minute_feature():
    dates = pd.DataFrame(
        {"date": pd.date_range("2021-01-01 00:00:00", periods=2, freq="15min")}
    )

    encoder = TimeFeatureEncoder(mode="calendar", freq="t")
    encoded = encoder.fit_transform(dates)

    assert encoded.shape == (2, 5)
    np.testing.assert_array_equal(encoded[:, -1], np.array([0, 15]))


def test_encoder_can_infer_frequency_from_dataframe_dates():
    dates = pd.DataFrame(
        {"date": pd.date_range("2021-01-01", periods=4, freq="h")}
    )

    encoder = TimeFeatureEncoder(mode="continuous", freq="infer")
    encoded = encoder.fit_transform(dates)

    assert encoded.shape == (4, 4)
    assert encoder.resolved_freq == "h"


def test_none_mode_returns_none_and_zero_dim():
    dates = pd.DataFrame(
        {"date": pd.date_range("2021-01-01", periods=4, freq="h")}
    )

    encoder = TimeFeatureEncoder(mode="none", freq="infer")

    assert encoder.fit_transform(dates) is None
    assert encoder.output_dim == 0
```

- [ ] **Step 2: Run time feature tests and verify they fail**

Run:

```bash
python -m pytest tests/dataloader/v2/test_time_features.py -q
```

Expected: FAIL because `time_features.py` does not exist.

- [ ] **Step 3: Implement TimeFeatureEncoder**

Create `torch_timeseries/dataloader/v2/time_features.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from torch_timeseries.utils.timefeatures import time_features


@dataclass
class TimeFeatureEncoder:
    mode: str = "continuous"
    freq: str | None = "infer"

    def __post_init__(self) -> None:
        allowed_modes = {"continuous", "calendar", "none"}
        if self.mode not in allowed_modes:
            raise ValueError(f"mode must be one of {sorted(allowed_modes)}")
        self.output_dim: int = 0
        self.resolved_freq: str | None = None

    def fit(self, dates: pd.DataFrame) -> "TimeFeatureEncoder":
        if self.mode == "none":
            self.output_dim = 0
            self.resolved_freq = None
            return self

        resolved_freq = self._resolve_freq(dates)
        encoded = self.transform(dates, resolved_freq=resolved_freq)
        self.resolved_freq = resolved_freq
        self.output_dim = encoded.shape[1]
        return self

    def transform(
        self,
        dates: pd.DataFrame,
        resolved_freq: str | None = None,
    ) -> np.ndarray | None:
        if self.mode == "none":
            return None

        freq = resolved_freq or self.resolved_freq or self._resolve_freq(dates)
        timeenc = 1 if self.mode == "continuous" else 0
        return time_features(dates.copy(), timeenc=timeenc, freq=freq)

    def fit_transform(self, dates: pd.DataFrame) -> np.ndarray | None:
        self.fit(dates)
        return self.transform(dates)

    def _resolve_freq(self, dates: pd.DataFrame) -> str:
        if self.freq and self.freq != "infer":
            return self.freq

        index = pd.DatetimeIndex(pd.to_datetime(dates["date"]))
        inferred = pd.infer_freq(index)
        if inferred is None:
            raise ValueError("Could not infer datetime frequency; pass freq explicitly")
        return _normalize_freq_alias(inferred)


def _normalize_freq_alias(freq: str) -> str:
    lowered = freq.lower()
    if lowered.endswith("min"):
        return "t"
    first = lowered[-1]
    aliases = {"h": "h", "d": "d", "b": "b", "w": "w", "m": "m", "s": "s", "t": "t"}
    return aliases.get(first, first)
```

- [ ] **Step 4: Export TimeFeatureEncoder**

Modify `torch_timeseries/dataloader/v2/__init__.py`:

```python
from .features import normalize_feature_names, resolve_feature_selection
from .scalers import ScalerBundle
from .splits import FixedBoundarySplit, RatioSplit, SplitIndices, SplitStrategy
from .time_features import TimeFeatureEncoder
from .typing import FeatureSelection, FeatureSelector

__all__ = [
    "FeatureSelection",
    "FeatureSelector",
    "FixedBoundarySplit",
    "RatioSplit",
    "ScalerBundle",
    "SplitIndices",
    "SplitStrategy",
    "TimeFeatureEncoder",
    "normalize_feature_names",
    "resolve_feature_selection",
]
```

- [ ] **Step 5: Run time feature tests**

Run:

```bash
python -m pytest tests/dataloader/v2/test_time_features.py -q
```

Expected: PASS. If `WeekOfYear` failures appear on newer pandas in nearby tests, fix `torch_timeseries/utils/timefeatures.py` in Task 11, not here.

- [ ] **Step 6: Commit**

```bash
git add torch_timeseries/dataloader/v2 tests/dataloader/v2/test_time_features.py
git commit -m "feat: add v2 time feature encoder"
```

---

## Task 5: WindowedDataset Dict Batch Contract

**Files:**
- Create: `torch_timeseries/dataloader/v2/window.py`
- Modify: `torch_timeseries/dataloader/v2/__init__.py`
- Test: `tests/dataloader/v2/test_window.py`

- [ ] **Step 1: Write failing window tests**

Create `tests/dataloader/v2/test_window.py`:

```python
import numpy as np
import pandas as pd

from torch_timeseries.dataloader.v2.scalers import ScalerBundle
from torch_timeseries.dataloader.v2.time_features import TimeFeatureEncoder
from torch_timeseries.dataloader.v2.window import WindowSpec, WindowedDataset
from torch_timeseries.scaler import StandardScaler


def _values():
    return np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
            [5.0, 50.0, 500.0],
            [6.0, 60.0, 600.0],
        ],
        dtype=np.float32,
    )


def _dates():
    return pd.DataFrame({"date": pd.date_range("2021-01-01", periods=6, freq="h")})


def test_windowed_dataset_supports_m_to_one_with_raw_and_time_fields():
    values = _values()
    scalers = ScalerBundle(StandardScaler)
    scalers.fit(values[:4], input_indices=[0, 1, 2], target_indices=[2])
    time_encoder = TimeFeatureEncoder(mode="continuous", freq="h")
    time_values = time_encoder.fit_transform(_dates())

    dataset = WindowedDataset(
        values=values,
        input_indices=[0, 1, 2],
        target_indices=[2],
        spec=WindowSpec(context_length=3, prediction_length=2, horizon=1),
        scalers=scalers,
        time_features=time_values,
        return_raw=True,
        time_index=np.arange(len(values)),
    )

    sample = dataset[0]

    assert set(sample) == {
        "x",
        "y",
        "raw_x",
        "raw_y",
        "x_time",
        "y_time",
        "x_index",
        "y_index",
    }
    assert sample["x"].shape == (3, 3)
    assert sample["y"].shape == (2, 1)
    assert sample["raw_y"].tolist() == [[300.0], [400.0]]
    assert sample["x_time"].shape == (3, 4)
    assert sample["y_time"].shape == (2, 4)
    assert sample["x_index"].tolist() == [0, 1, 2]
    assert sample["y_index"].tolist() == [2, 3]


def test_windowed_dataset_supports_m_to_k():
    values = _values()
    scalers = ScalerBundle(StandardScaler)
    scalers.fit(values[:4], input_indices=[0, 1, 2], target_indices=[0, 2])

    dataset = WindowedDataset(
        values=values,
        input_indices=[0, 1, 2],
        target_indices=[0, 2],
        spec=WindowSpec(context_length=2, prediction_length=3, horizon=1),
        scalers=scalers,
        return_raw=False,
    )

    sample = dataset[1]

    assert sample["x"].shape == (2, 3)
    assert sample["y"].shape == (3, 2)
    assert "raw_x" not in sample
    assert "raw_y" not in sample


def test_window_stride_can_represent_non_overlap_fast_mode():
    values = _values()
    scalers = ScalerBundle(StandardScaler)
    scalers.fit(values[:4], input_indices=[0, 1, 2], target_indices=[0, 1, 2])

    dataset = WindowedDataset(
        values=values,
        input_indices=[0, 1, 2],
        target_indices=[0, 1, 2],
        spec=WindowSpec(context_length=2, prediction_length=1, horizon=1, stride=3),
        scalers=scalers,
    )

    assert len(dataset) == 2
    assert dataset[1]["x"].shape == (2, 3)
```

- [ ] **Step 2: Run window tests and verify they fail**

Run:

```bash
python -m pytest tests/dataloader/v2/test_window.py -q
```

Expected: FAIL because `window.py` does not exist.

- [ ] **Step 3: Implement WindowSpec and WindowedDataset**

Create `torch_timeseries/dataloader/v2/window.py`:

```python
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset

from .scalers import ScalerBundle


@dataclass(frozen=True)
class WindowSpec:
    context_length: int
    prediction_length: int
    horizon: int = 1
    stride: int = 1

    def __post_init__(self) -> None:
        for name in ("context_length", "prediction_length", "horizon", "stride"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def required_length(self) -> int:
        return self.context_length + self.horizon - 1 + self.prediction_length


class WindowedDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        input_indices: Sequence[int],
        target_indices: Sequence[int],
        spec: WindowSpec,
        scalers: ScalerBundle,
        time_features: np.ndarray | None = None,
        return_raw: bool = False,
        time_index: np.ndarray | None = None,
    ) -> None:
        if values.ndim != 2:
            raise ValueError("values must have shape [time, features]")
        if len(values) < spec.required_length:
            raise ValueError(
                f"Dataset length {len(values)} is shorter than required window length "
                f"{spec.required_length}"
            )
        if time_features is not None and len(time_features) != len(values):
            raise ValueError("time_features length must match values length")
        if time_index is not None and len(time_index) != len(values):
            raise ValueError("time_index length must match values length")

        self.values = values
        self.input_indices = list(input_indices)
        self.target_indices = list(target_indices)
        self.spec = spec
        self.scalers = scalers
        self.time_features = time_features
        self.return_raw = return_raw
        self.time_index = time_index

    def __len__(self) -> int:
        usable = len(self.values) - self.spec.required_length
        return usable // self.spec.stride + 1

    def __getitem__(self, item: int) -> dict[str, np.ndarray]:
        if not isinstance(item, int):
            raise TypeError("WindowedDataset only supports integer indexing")

        start = item * self.spec.stride
        x_start = start
        x_end = x_start + self.spec.context_length
        y_start = x_start + self.spec.context_length + self.spec.horizon - 1
        y_end = y_start + self.spec.prediction_length

        raw_x = self.values[x_start:x_end, :][:, self.input_indices]
        raw_y = self.values[y_start:y_end, :][:, self.target_indices]

        sample: dict[str, np.ndarray] = {
            "x": self.scalers.transform_input(raw_x).astype(np.float32),
            "y": self.scalers.transform_target(raw_y).astype(np.float32),
        }

        if self.return_raw:
            sample["raw_x"] = raw_x.astype(np.float32)
            sample["raw_y"] = raw_y.astype(np.float32)

        if self.time_features is not None:
            sample["x_time"] = self.time_features[x_start:x_end].astype(np.float32)
            sample["y_time"] = self.time_features[y_start:y_end].astype(np.float32)

        if self.time_index is not None:
            sample["x_index"] = self.time_index[x_start:x_end]
            sample["y_index"] = self.time_index[y_start:y_end]

        return sample
```

- [ ] **Step 4: Export window symbols**

Modify `torch_timeseries/dataloader/v2/__init__.py`:

```python
from .features import normalize_feature_names, resolve_feature_selection
from .scalers import ScalerBundle
from .splits import FixedBoundarySplit, RatioSplit, SplitIndices, SplitStrategy
from .time_features import TimeFeatureEncoder
from .typing import FeatureSelection, FeatureSelector
from .window import WindowedDataset, WindowSpec

__all__ = [
    "FeatureSelection",
    "FeatureSelector",
    "FixedBoundarySplit",
    "RatioSplit",
    "ScalerBundle",
    "SplitIndices",
    "SplitStrategy",
    "TimeFeatureEncoder",
    "WindowSpec",
    "WindowedDataset",
    "normalize_feature_names",
    "resolve_feature_selection",
]
```

- [ ] **Step 5: Run window tests**

Run:

```bash
python -m pytest tests/dataloader/v2/test_window.py -q
```

Expected: PASS.

- [ ] **Step 6: Run all v2 unit tests so far**

Run:

```bash
python -m pytest tests/dataloader/v2 -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add torch_timeseries/dataloader/v2 tests/dataloader/v2/test_window.py
git commit -m "feat: add v2 windowed dataset"
```

---

## Task 6: TimeSeriesDataModule Orchestration

**Files:**
- Create: `torch_timeseries/dataloader/v2/module.py`
- Modify: `torch_timeseries/dataloader/v2/__init__.py`
- Test: `tests/dataloader/v2/test_module.py`

- [ ] **Step 1: Write failing data module tests**

Create `tests/dataloader/v2/test_module.py`:

```python
import numpy as np
import pandas as pd

from torch_timeseries.dataloader.v2.module import TimeSeriesDataModule
from torch_timeseries.dataloader.v2.splits import RatioSplit
from torch_timeseries.dataloader.v2.window import WindowSpec
from torch_timeseries.scaler import StandardScaler


class ArrayDataset:
    name = "array"
    freq = "h"

    def __init__(self):
        self.data = np.arange(60, dtype=np.float32).reshape(20, 3)
        self.num_features = 3
        self.dates = pd.DataFrame(
            {"date": pd.date_range("2021-01-01", periods=20, freq="h")}
        )
        self.time_index = np.arange(20)

    def __len__(self):
        return len(self.data)


def test_data_module_builds_dict_batches_for_m_to_one():
    module = TimeSeriesDataModule(
        dataset=ArrayDataset(),
        window=WindowSpec(context_length=4, prediction_length=2, horizon=1),
        split=RatioSplit(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2),
        scaler_factory=StandardScaler,
        input_features=None,
        target_features=[-1],
        batch_size=2,
        time_feature_mode="continuous",
        return_raw=True,
        return_time_index=True,
        num_workers=0,
    )

    batch = next(iter(module.train_loader))

    assert batch["x"].shape == (2, 4, 3)
    assert batch["y"].shape == (2, 2, 1)
    assert batch["raw_y"].shape == (2, 2, 1)
    assert batch["x_time"].shape == (2, 4, 4)
    assert batch["y_time"].shape == (2, 2, 4)
    assert batch["x_index"].shape == (2, 4)
    assert module.input_selection.indices == [0, 1, 2]
    assert module.target_selection.indices == [2]
    assert module.time_feature_dim == 4


def test_data_module_defaults_target_to_all_features():
    module = TimeSeriesDataModule(
        dataset=ArrayDataset(),
        window=WindowSpec(context_length=4, prediction_length=2, horizon=1),
        split=RatioSplit(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2),
        scaler_factory=StandardScaler,
        batch_size=2,
        time_feature_mode="none",
        num_workers=0,
    )

    batch = next(iter(module.train_loader))

    assert batch["x"].shape[-1] == 3
    assert batch["y"].shape[-1] == 3
    assert "x_time" not in batch
```

- [ ] **Step 2: Run data module tests and verify they fail**

Run:

```bash
python -m pytest tests/dataloader/v2/test_module.py -q
```

Expected: FAIL because `module.py` does not exist.

- [ ] **Step 3: Implement TimeSeriesDataModule**

Create `torch_timeseries/dataloader/v2/module.py`:

```python
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from torch.utils.data import DataLoader

from torch_timeseries.dataloader._seed import seed_worker
from torch_timeseries.scaler import StandardScaler

from .features import normalize_feature_names, resolve_feature_selection
from .scalers import ScalerBundle
from .splits import RatioSplit, SplitStrategy
from .time_features import TimeFeatureEncoder
from .typing import FeatureSelection, FeatureSelector
from .window import WindowedDataset, WindowSpec


class TimeSeriesDataModule:
    def __init__(
        self,
        dataset,
        window: WindowSpec,
        split: SplitStrategy | None = None,
        scaler_factory=StandardScaler,
        input_features: FeatureSelector = None,
        target_features: FeatureSelector = None,
        feature_names: Sequence[str] | None = None,
        batch_size: int = 32,
        shuffle_train: bool = True,
        num_workers: int = 0,
        time_feature_mode: str = "continuous",
        time_feature_freq: str | None = None,
        return_raw: bool = False,
        return_time_index: bool = False,
    ) -> None:
        self.dataset = dataset
        self.window = window
        self.split_strategy = split or RatioSplit()
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers
        self.return_raw = return_raw
        self.return_time_index = return_time_index

        values = np.asarray(dataset.data)
        if values.ndim != 2:
            raise ValueError("dataset.data must have shape [time, features]")
        self.values = values
        self.feature_names = normalize_feature_names(
            values.shape[1],
            feature_names or _dataset_feature_names(dataset),
        )
        self.input_selection = resolve_feature_selection(
            input_features,
            values.shape[1],
            self.feature_names,
            role="input_features",
        )
        self.target_selection = resolve_feature_selection(
            target_features,
            values.shape[1],
            self.feature_names,
            role="target_features",
        )

        self.split_indices = self.split_strategy.split(
            length=len(values),
            context_length=window.context_length,
            horizon=window.horizon,
            prediction_length=window.prediction_length,
        )

        self.scalers = ScalerBundle(scaler_factory)
        self.scalers.fit(
            values[self.split_indices.train],
            input_indices=self.input_selection.indices,
            target_indices=self.target_selection.indices,
        )

        freq = time_feature_freq
        if freq is None:
            freq = getattr(dataset, "freq", "infer")
        self.time_encoder = TimeFeatureEncoder(mode=time_feature_mode, freq=freq)
        self.time_features = self.time_encoder.fit_transform(dataset.dates)
        self.time_feature_dim = self.time_encoder.output_dim

        full_time_index = getattr(dataset, "time_index", None)
        self.train_dataset = self._make_windowed_dataset(self.split_indices.train, full_time_index)
        self.val_dataset = self._make_windowed_dataset(self.split_indices.val, full_time_index)
        self.test_dataset = self._make_windowed_dataset(self.split_indices.test, full_time_index)

        self.train_loader = self._make_loader(self.train_dataset, shuffle=self.shuffle_train)
        self.val_loader = self._make_loader(self.val_dataset, shuffle=False)
        self.test_loader = self._make_loader(self.test_dataset, shuffle=False)

    @property
    def enc_in(self) -> int:
        return len(self.input_selection.indices)

    @property
    def c_out(self) -> int:
        return len(self.target_selection.indices)

    def _make_windowed_dataset(
        self,
        indices: list[int],
        full_time_index: np.ndarray | None,
    ) -> WindowedDataset:
        values = self.values[indices]
        time_features = None if self.time_features is None else self.time_features[indices]
        time_index = None
        if self.return_time_index:
            if full_time_index is None:
                time_index = np.asarray(indices)
            else:
                time_index = np.asarray(full_time_index)[indices]

        return WindowedDataset(
            values=values,
            input_indices=self.input_selection.indices,
            target_indices=self.target_selection.indices,
            spec=self.window,
            scalers=self.scalers,
            time_features=time_features,
            return_raw=self.return_raw,
            time_index=time_index,
        )

    def _make_loader(self, dataset: WindowedDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
        )


def _dataset_feature_names(dataset) -> Sequence[str] | None:
    if hasattr(dataset, "feature_names"):
        return dataset.feature_names
    if hasattr(dataset, "df") and "date" in dataset.df.columns:
        return [col for col in dataset.df.columns if col != "date"]
    return None
```

- [ ] **Step 4: Export TimeSeriesDataModule**

Modify `torch_timeseries/dataloader/v2/__init__.py`:

```python
from .features import normalize_feature_names, resolve_feature_selection
from .module import TimeSeriesDataModule
from .scalers import ScalerBundle
from .splits import FixedBoundarySplit, RatioSplit, SplitIndices, SplitStrategy
from .time_features import TimeFeatureEncoder
from .typing import FeatureSelection, FeatureSelector
from .window import WindowedDataset, WindowSpec

__all__ = [
    "FeatureSelection",
    "FeatureSelector",
    "FixedBoundarySplit",
    "RatioSplit",
    "ScalerBundle",
    "SplitIndices",
    "SplitStrategy",
    "TimeFeatureEncoder",
    "TimeSeriesDataModule",
    "WindowSpec",
    "WindowedDataset",
    "normalize_feature_names",
    "resolve_feature_selection",
]
```

- [ ] **Step 5: Run data module tests**

Run:

```bash
python -m pytest tests/dataloader/v2/test_module.py -q
```

Expected: PASS.

- [ ] **Step 6: Run all v2 unit tests**

Run:

```bash
python -m pytest tests/dataloader/v2 -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add torch_timeseries/dataloader/v2 tests/dataloader/v2/test_module.py
git commit -m "feat: add v2 time series data module"
```

---

## Task 7: Legacy Tuple Adapter

**Files:**
- Create: `torch_timeseries/dataloader/v2/adapters.py`
- Modify: `torch_timeseries/dataloader/v2/__init__.py`
- Test: `tests/dataloader/v2/test_adapters.py`

- [ ] **Step 1: Write failing legacy adapter tests**

Create `tests/dataloader/v2/test_adapters.py`:

```python
import numpy as np
import pandas as pd

from torch_timeseries.dataloader.v2.adapters import LegacyForecastTupleDataset
from torch_timeseries.dataloader.v2.module import TimeSeriesDataModule
from torch_timeseries.dataloader.v2.splits import RatioSplit
from torch_timeseries.dataloader.v2.window import WindowSpec
from torch_timeseries.scaler import StandardScaler


class ArrayDataset:
    name = "array"
    freq = "h"

    def __init__(self):
        self.data = np.arange(90, dtype=np.float32).reshape(30, 3)
        self.num_features = 3
        self.dates = pd.DataFrame(
            {"date": pd.date_range("2021-01-01", periods=30, freq="h")}
        )
        self.time_index = np.arange(30)

    def __len__(self):
        return len(self.data)


def test_legacy_adapter_returns_six_item_tuple_with_raw():
    module = TimeSeriesDataModule(
        dataset=ArrayDataset(),
        window=WindowSpec(context_length=4, prediction_length=2, horizon=1),
        split=RatioSplit(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2),
        scaler_factory=StandardScaler,
        batch_size=2,
        return_raw=True,
        num_workers=0,
    )
    dataset = LegacyForecastTupleDataset(module.train_dataset, include_raw=True)

    sample = dataset[0]

    assert len(sample) == 6
    assert sample[0].shape == (4, 3)
    assert sample[1].shape == (2, 3)
    assert sample[2].shape == (4, 3)
    assert sample[3].shape == (2, 3)
    assert sample[4].shape == (4, 4)
    assert sample[5].shape == (2, 4)


def test_legacy_adapter_can_append_time_index():
    module = TimeSeriesDataModule(
        dataset=ArrayDataset(),
        window=WindowSpec(context_length=4, prediction_length=2, horizon=1),
        split=RatioSplit(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2),
        scaler_factory=StandardScaler,
        batch_size=2,
        return_raw=True,
        return_time_index=True,
        num_workers=0,
    )
    dataset = LegacyForecastTupleDataset(
        module.train_dataset,
        include_raw=True,
        time_index=True,
    )

    sample = dataset[0]

    assert len(sample) == 8
    assert sample[6].shape == (4,)
    assert sample[7].shape == (2,)
```

- [ ] **Step 2: Run adapter tests and verify they fail**

Run:

```bash
python -m pytest tests/dataloader/v2/test_adapters.py -q
```

Expected: FAIL because `adapters.py` does not exist.

- [ ] **Step 3: Implement legacy tuple adapter**

Create `torch_timeseries/dataloader/v2/adapters.py`:

```python
from __future__ import annotations

from torch.utils.data import Dataset


class LegacyForecastTupleDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        include_raw: bool = True,
        time_index: bool = False,
    ) -> None:
        self.dataset = dataset
        self.include_raw = include_raw
        self.time_index = time_index

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]

        result = [sample["x"], sample["y"]]
        if self.include_raw:
            result.extend([sample["raw_x"], sample["raw_y"]])
        result.extend([sample.get("x_time"), sample.get("y_time")])
        if self.time_index:
            result.extend([sample["x_index"], sample["y_index"]])
        return tuple(result)
```

- [ ] **Step 4: Export adapter**

Modify `torch_timeseries/dataloader/v2/__init__.py`:

```python
from .adapters import LegacyForecastTupleDataset
from .features import normalize_feature_names, resolve_feature_selection
from .module import TimeSeriesDataModule
from .scalers import ScalerBundle
from .splits import FixedBoundarySplit, RatioSplit, SplitIndices, SplitStrategy
from .time_features import TimeFeatureEncoder
from .typing import FeatureSelection, FeatureSelector
from .window import WindowedDataset, WindowSpec

__all__ = [
    "FeatureSelection",
    "FeatureSelector",
    "FixedBoundarySplit",
    "LegacyForecastTupleDataset",
    "RatioSplit",
    "ScalerBundle",
    "SplitIndices",
    "SplitStrategy",
    "TimeFeatureEncoder",
    "TimeSeriesDataModule",
    "WindowSpec",
    "WindowedDataset",
    "normalize_feature_names",
    "resolve_feature_selection",
]
```

- [ ] **Step 5: Run adapter tests**

Run:

```bash
python -m pytest tests/dataloader/v2/test_adapters.py -q
```

Expected: PASS.

- [ ] **Step 6: Run all v2 tests**

Run:

```bash
python -m pytest tests/dataloader/v2 -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add torch_timeseries/dataloader/v2 tests/dataloader/v2/test_adapters.py
git commit -m "feat: add legacy forecast tuple adapter"
```

---

## Task 8: Compatibility Loader For SlidingWindowTS

**Files:**
- Modify: `torch_timeseries/dataloader/sliding_window_ts.py`
- Modify: `torch_timeseries/dataloader/ETT.py`
- Test: `tests/dataloader/v2/test_legacy_sliding_window_ts.py`
- Test: existing `tests/dataloader/slidingwindowts.py`

- [ ] **Step 1: Write failing compatibility tests**

Create `tests/dataloader/v2/test_legacy_sliding_window_ts.py`:

```python
import numpy as np
import pandas as pd

from torch_timeseries.dataloader import SlidingWindowTS
from torch_timeseries.scaler import StandardScaler


class ArrayDataset:
    name = "array"
    freq = "h"

    def __init__(self):
        self.data = np.arange(120, dtype=np.float32).reshape(40, 3)
        self.num_features = 3
        self.df = pd.DataFrame(self.data, columns=["a", "b", "c"])
        self.df.insert(0, "date", pd.date_range("2021-01-01", periods=40, freq="h"))
        self.dates = pd.DataFrame({"date": self.df["date"]})
        self.time_index = np.arange(40)

    def __len__(self):
        return len(self.data)


def test_sliding_window_ts_accepts_target_features_for_m_to_one():
    loader = SlidingWindowTS(
        ArrayDataset(),
        scaler=StandardScaler(),
        window=4,
        horizon=1,
        steps=2,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        batch_size=2,
        num_worker=0,
        freq="h",
        target_features=[-1],
    )

    batch = next(iter(loader.train_loader))

    assert len(batch) == 6
    assert batch[0].shape == (2, 4, 3)
    assert batch[1].shape == (2, 2, 1)
    assert batch[3].shape == (2, 2, 1)


def test_sliding_window_ts_keeps_old_default_m_to_m_behavior():
    loader = SlidingWindowTS(
        ArrayDataset(),
        scaler=StandardScaler(),
        window=4,
        horizon=1,
        steps=2,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        batch_size=2,
        num_worker=0,
        freq="h",
    )

    batch = next(iter(loader.train_loader))

    assert batch[0].shape == (2, 4, 3)
    assert batch[1].shape == (2, 2, 3)
```

- [ ] **Step 2: Run compatibility tests and verify the new selector test fails**

Run:

```bash
python -m pytest tests/dataloader/v2/test_legacy_sliding_window_ts.py -q
```

Expected: FAIL because `SlidingWindowTS.__init__` does not accept `target_features`.

- [ ] **Step 3: Reimplement SlidingWindowTS through v2 while preserving public attributes**

Modify `torch_timeseries/dataloader/sliding_window_ts.py` by replacing internal split/wrapper creation with `TimeSeriesDataModule` and `LegacyForecastTupleDataset`. Keep the public constructor name and existing parameter names. Add new optional parameters at the end:

```python
input_features=None,
target_features=None,
feature_names=None,
```

Inside `__init__`, after assigning existing fields, build:

```python
from torch.utils.data import DataLoader

from .v2 import (
    LegacyForecastTupleDataset,
    RatioSplit,
    TimeSeriesDataModule,
    WindowSpec,
)
from ._seed import seed_worker

window_spec = WindowSpec(
    context_length=self.window,
    prediction_length=self.steps,
    horizon=self.horizon,
    stride=(self.window + self.horizon + self.steps - 1) if self.fast_train else 1,
)

self._v2_module = TimeSeriesDataModule(
    dataset=self.dataset,
    window=window_spec,
    split=RatioSplit(
        train_ratio=self.train_ratio,
        val_ratio=self.val_ratio,
        test_ratio=self.test_ratio,
        uniform_eval=self.uniform_eval,
    ),
    scaler_factory=lambda: self.scaler,
    input_features=input_features,
    target_features=target_features,
    feature_names=feature_names,
    batch_size=self.batch_size,
    shuffle_train=self.shuffle_train,
    num_workers=self.num_worker,
    time_feature_mode="continuous" if self.time_enc == 1 else "calendar",
    time_feature_freq=self.freq,
    return_raw=self.include_raw,
    return_time_index=self.time_index,
)
self.scaler = self._v2_module.scalers.target_scaler
self.train_dataset = LegacyForecastTupleDataset(
    self._v2_module.train_dataset,
    include_raw=self.include_raw,
    time_index=self.time_index,
)
self.val_dataset = LegacyForecastTupleDataset(
    self._v2_module.val_dataset,
    include_raw=self.include_raw,
    time_index=self.time_index,
)
self.test_dataset = LegacyForecastTupleDataset(
    self._v2_module.test_dataset,
    include_raw=self.include_raw,
    time_index=self.time_index,
)
self.train_loader = DataLoader(
    self.train_dataset,
    batch_size=self.batch_size,
    shuffle=self.shuffle_train,
    num_workers=self.num_worker,
    worker_init_fn=seed_worker,
)
self.val_loader = DataLoader(
    self.val_dataset,
    batch_size=self.batch_size,
    shuffle=False,
    num_workers=self.num_worker,
    worker_init_fn=seed_worker,
)
self.test_loader = DataLoader(
    self.test_dataset,
    batch_size=self.batch_size,
    shuffle=False,
    num_workers=self.num_worker,
    worker_init_fn=seed_worker,
)
self.train_size = len(self.train_dataset)
self.val_size = len(self.val_dataset)
self.test_size = len(self.test_dataset)
```

When handling `fast_val` and `fast_test`, create separate `WindowSpec` instances for val/test with non-overlap stride only for the requested split. If this makes the initial patch large, leave `fast_*` behavior unchanged by keeping old code path when any `fast_*` flag is true, and only route to v2 when all `fast_*` flags are false. Add a comment explaining that Task 9 removes this fallback.

- [ ] **Step 4: Run the compatibility tests**

Run:

```bash
python -m pytest tests/dataloader/v2/test_legacy_sliding_window_ts.py -q
```

Expected: PASS.

- [ ] **Step 5: Run existing sliding window tests**

Run:

```bash
python -m pytest tests/dataloader/slidingwindowts.py -q
```

Expected: PASS. Existing tuple shapes remain unchanged for default M->M and `single_variate=True` old behavior if the old path is retained for `single_variate=True`.

- [ ] **Step 6: Commit**

```bash
git add torch_timeseries/dataloader/sliding_window_ts.py tests/dataloader/v2/test_legacy_sliding_window_ts.py
git commit -m "feat: route sliding window loader through v2"
```

---

## Task 9: ETT Fixed Boundary Split Without Loader Subclass Logic

**Files:**
- Modify: `torch_timeseries/dataloader/ETT.py`
- Test: `tests/dataloader/v2/test_ett_split.py`

- [ ] **Step 1: Write failing ETT split tests with small synthetic boundaries**

Create `tests/dataloader/v2/test_ett_split.py`:

```python
from torch_timeseries.dataloader.v2.splits import FixedBoundarySplit


def test_etth_boundary_formula_can_be_represented_as_fixed_split():
    window = 4
    horizon = 1
    border1s = [0, 12 - window - horizon + 1, 16 - window - horizon + 1]
    border2s = [12, 16, 20]

    split = FixedBoundarySplit(
        train=(border1s[0], border2s[0]),
        val=(border1s[1], border2s[1]),
        test=(border1s[2], border2s[2]),
    )
    indices = split.split(length=20, context_length=window, horizon=horizon, prediction_length=2)

    assert indices.train[0] == 0
    assert indices.train[-1] == 11
    assert indices.val[0] == 8
    assert indices.test[0] == 12
```

- [ ] **Step 2: Run ETT split test**

Run:

```bash
python -m pytest tests/dataloader/v2/test_ett_split.py -q
```

Expected: PASS because `FixedBoundarySplit` already exists. This test documents the target behavior before changing ETT loader internals.

- [ ] **Step 3: Refactor ETTHLoader and ETTMLoader to supply FixedBoundarySplit**

Modify `torch_timeseries/dataloader/ETT.py` so `ETTHLoader` and `ETTMLoader` compute their current `border1s` and `border2s`, then pass `FixedBoundarySplit` into the same v2 compatibility construction used by `SlidingWindowTS`.

Add helper methods:

```python
def _split_strategy(self):
    border1s = [
        0,
        12 * 30 * 24 - self.window - self.horizon + 1,
        12 * 30 * 24 + 4 * 30 * 24 - self.window - self.horizon + 1,
    ]
    border2s = [
        12 * 30 * 24,
        12 * 30 * 24 + 4 * 30 * 24,
        12 * 30 * 24 + 8 * 30 * 24,
    ]
    return FixedBoundarySplit(
        train=(border1s[0], border2s[0]),
        val=(border1s[1], border2s[1]),
        test=(border1s[2], border2s[2]),
    )
```

For `ETTMLoader`, multiply the hourly constants by `4` as in the current code.

- [ ] **Step 4: Run ETT and sliding window tests**

Run:

```bash
python -m pytest tests/dataloader/v2/test_ett_split.py tests/dataloader/v2/test_legacy_sliding_window_ts.py tests/dataloader/slidingwindowts.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add torch_timeseries/dataloader/ETT.py tests/dataloader/v2/test_ett_split.py
git commit -m "refactor: express ETT loaders with fixed split strategy"
```

---

## Task 10: Custom Dataset Constructors For CSV/DataFrame/Array

**Files:**
- Modify: `torch_timeseries/core/dataset/dataset.py`
- Test: `tests/dataloader/v2/test_custom_dataset.py`

- [ ] **Step 1: Write failing custom dataset constructor tests**

Create `tests/dataloader/v2/test_custom_dataset.py`:

```python
import numpy as np
import pandas as pd
import pytest

from torch_timeseries.core import TimeSeriesDataset


def test_from_dataframe_builds_dataset_with_names_and_inferred_freq():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=4, freq="h"),
            "load": [1.0, 2.0, 3.0, 4.0],
            "temp": [10.0, 11.0, 12.0, 13.0],
        }
    )

    dataset = TimeSeriesDataset.from_dataframe(
        df,
        time_col="date",
        target_cols=["load"],
        freq="infer",
        name="custom",
    )

    assert dataset.name == "custom"
    assert dataset.freq == "h"
    assert dataset.feature_names == ["load", "temp"]
    assert dataset.data.shape == (4, 2)
    assert dataset.target_cols == ["load"]


def test_from_array_generates_feature_names():
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    dates = pd.date_range("2021-01-01", periods=4, freq="d")

    dataset = TimeSeriesDataset.from_array(
        values,
        datetime_index=dates,
        freq="infer",
        name="array",
    )

    assert dataset.freq == "d"
    assert dataset.feature_names == ["feature_0", "feature_1", "feature_2"]
    assert dataset.num_features == 3


def test_from_dataframe_rejects_duplicate_timestamps():
    df = pd.DataFrame(
        {
            "date": ["2021-01-01", "2021-01-01"],
            "value": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        TimeSeriesDataset.from_dataframe(df, time_col="date")
```

- [ ] **Step 2: Run custom dataset tests and verify they fail**

Run:

```bash
python -m pytest tests/dataloader/v2/test_custom_dataset.py -q
```

Expected: FAIL because constructors do not exist.

- [ ] **Step 3: Add factory constructors without breaking existing subclasses**

Modify `torch_timeseries/core/dataset/dataset.py`.

Add a concrete lightweight class near `TimeSeriesDataset`:

```python
class InMemoryTimeSeriesDataset(TimeSeriesDataset):
    def __init__(
        self,
        data: np.ndarray,
        dates: pd.DataFrame,
        df: pd.DataFrame,
        name: str = "custom",
        freq: str | None = None,
        feature_names: Sequence[str] | None = None,
        target_cols: Sequence[str] | None = None,
    ):
        torch.utils.data.Dataset.__init__(self)
        self.root = ""
        self.dir = ""
        self.name = name
        self.data = data
        self.dates = dates
        self.df = df
        self.freq = freq
        self.feature_names = list(feature_names or [f"feature_{i}" for i in range(data.shape[1])])
        self.target_cols = list(target_cols or [])
        self.num_features = data.shape[1]
        self.length = data.shape[0]
        self.time_index = np.arange(self.length)

    def download(self):
        return None

    def _load(self):
        return self.data
```

Add classmethods on `TimeSeriesDataset`:

```python
@classmethod
def from_dataframe(
    cls,
    df: pd.DataFrame,
    time_col: str = "date",
    target_cols: Sequence[str] | None = None,
    feature_cols: Sequence[str] | None = None,
    freq: str | None = "infer",
    name: str = "custom",
):
    frame = df.copy()
    frame[time_col] = pd.to_datetime(frame[time_col])
    if frame[time_col].duplicated().any():
        raise ValueError("duplicate timestamps are not supported")
    if not frame[time_col].is_monotonic_increasing:
        frame = frame.sort_values(time_col).reset_index(drop=True)

    if feature_cols is None:
        feature_cols = [col for col in frame.columns if col != time_col]
    feature_cols = list(feature_cols)
    non_numeric = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(frame[col])]
    if non_numeric:
        raise ValueError(f"feature columns must be numeric: {non_numeric}")

    resolved_freq = _resolve_dataset_freq(frame[time_col], freq)
    values = frame[feature_cols].to_numpy()
    out_df = frame[[time_col] + feature_cols].rename(columns={time_col: "date"})
    dates = pd.DataFrame({"date": out_df["date"]})
    return InMemoryTimeSeriesDataset(
        data=values,
        dates=dates,
        df=out_df,
        name=name,
        freq=resolved_freq,
        feature_names=feature_cols,
        target_cols=target_cols,
    )

@classmethod
def from_array(
    cls,
    values: np.ndarray,
    datetime_index,
    feature_names: Sequence[str] | None = None,
    freq: str | None = "infer",
    name: str = "array",
):
    data = np.asarray(values)
    if data.ndim != 2:
        raise ValueError("values must have shape [time, features]")
    dates = pd.DataFrame({"date": pd.to_datetime(datetime_index)})
    frame = pd.DataFrame(data, columns=list(feature_names or [f"feature_{i}" for i in range(data.shape[1])]))
    frame.insert(0, "date", dates["date"])
    return cls.from_dataframe(
        frame,
        time_col="date",
        feature_cols=list(frame.columns[1:]),
        freq=freq,
        name=name,
    )
```

Add helper:

```python
def _resolve_dataset_freq(values, freq):
    if freq is None or freq != "infer":
        return freq
    inferred = pd.infer_freq(pd.DatetimeIndex(pd.to_datetime(values)))
    if inferred is None:
        raise ValueError("Could not infer datetime frequency; pass freq explicitly")
    lowered = inferred.lower()
    if lowered.endswith("min"):
        return "t"
    return lowered[-1]
```

- [ ] **Step 4: Export InMemoryTimeSeriesDataset**

Modify `torch_timeseries/core/dataset/__init__.py` and `torch_timeseries/core/__init__.py` to export `InMemoryTimeSeriesDataset`.

- [ ] **Step 5: Run custom dataset tests**

Run:

```bash
python -m pytest tests/dataloader/v2/test_custom_dataset.py -q
```

Expected: PASS.

- [ ] **Step 6: Run v2 tests**

Run:

```bash
python -m pytest tests/dataloader/v2 -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add torch_timeseries/core/dataset torch_timeseries/core/__init__.py tests/dataloader/v2/test_custom_dataset.py
git commit -m "feat: add in-memory time series dataset constructors"
```

---

## Task 11: Fix Legacy Time Feature Bugs And Decouple Embedding Input Dim

**Files:**
- Modify: `torch_timeseries/utils/timefeatures.py`
- Modify: `torch_timeseries/nn/embedding.py`
- Test: `tests/dataloader/v2/test_time_features_legacy.py`
- Test: `tests/nn/test_time_feature_embedding.py`

- [ ] **Step 1: Write failing tests for legacy time features**

Create `tests/dataloader/v2/test_time_features_legacy.py`:

```python
import pandas as pd

from torch_timeseries.utils.timefeatures import WeekOfYear, time_features


def test_time_features_does_not_mutate_input_dataframe():
    dates = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=2, freq="h")})
    before_columns = list(dates.columns)

    encoded = time_features(dates, timeenc=0, freq="h")

    assert encoded.shape == (2, 4)
    assert list(dates.columns) == before_columns


def test_week_of_year_uses_supported_pandas_api():
    index = pd.DatetimeIndex(pd.date_range("2021-01-01", periods=2, freq="w"))

    values = WeekOfYear()(index)

    assert values.shape == (2,)
```

- [ ] **Step 2: Write failing tests for embedding input_dim**

Create `tests/nn/test_time_feature_embedding.py`:

```python
import torch

from torch_timeseries.nn.embedding import DataEmbedding, TimeFeatureEmbedding


def test_time_feature_embedding_accepts_explicit_input_dim():
    embedding = TimeFeatureEmbedding(d_model=8, input_dim=6)
    x = torch.zeros(2, 4, 6)

    out = embedding(x)

    assert out.shape == (2, 4, 8)


def test_data_embedding_passes_time_feature_dim_for_timef():
    embedding = DataEmbedding(
        c_in=3,
        d_model=8,
        embed_type="timeF",
        freq="h",
        dropout=0.0,
        time_embed=True,
        time_feature_dim=6,
    )
    x = torch.zeros(2, 4, 3)
    x_mark = torch.zeros(2, 4, 6)

    out = embedding(x, x_mark)

    assert out.shape == (2, 4, 8)
```

- [ ] **Step 3: Run tests and verify failures**

Run:

```bash
python -m pytest tests/dataloader/v2/test_time_features_legacy.py tests/nn/test_time_feature_embedding.py -q
```

Expected: FAIL because `time_features` mutates input and embedding has no `input_dim`/`time_feature_dim` argument.

- [ ] **Step 4: Fix non-mutating time_features and WeekOfYear**

Modify `torch_timeseries/utils/timefeatures.py`:

```python
class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week.to_numpy() - 1) / 52.0 - 0.5
```

At the start of `time_features`, copy the input:

```python
def time_features(dates: pd.DataFrame, timeenc=0, freq='h') -> np.ndarray:
    dates = dates.copy()
```

Remove the duplicate `'yt'` key in `freq_map`.

- [ ] **Step 5: Add explicit input_dim to embedding classes**

Modify `torch_timeseries/nn/embedding.py`:

```python
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h', input_dim=None):
        super(TimeFeatureEmbedding, self).__init__()

        d_inp = input_dim if input_dim is not None else freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)
```

Modify `DataEmbedding.__init__`:

```python
def __init__(
    self,
    c_in,
    d_model,
    embed_type='fixed',
    freq='h',
    dropout=0.1,
    time_embed=True,
    time_feature_dim=None,
):
```

Use:

```python
self.temporal_embedding = (
    TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
    if embed_type != 'timeF'
    else TimeFeatureEmbedding(
        d_model=d_model,
        embed_type=embed_type,
        freq=freq,
        input_dim=time_feature_dim,
    )
)
```

Apply the same `time_feature_dim=None` argument to `DataEmbedding_wo_pos`.

- [ ] **Step 6: Run legacy time feature and embedding tests**

Run:

```bash
python -m pytest tests/dataloader/v2/test_time_features_legacy.py tests/nn/test_time_feature_embedding.py -q
```

Expected: PASS.

- [ ] **Step 7: Run all v2 and nn tests touched**

Run:

```bash
python -m pytest tests/dataloader/v2 tests/nn/test_time_feature_embedding.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add torch_timeseries/utils/timefeatures.py torch_timeseries/nn/embedding.py tests/dataloader/v2/test_time_features_legacy.py tests/nn/test_time_feature_embedding.py
git commit -m "fix: decouple time feature dimensions from frequency"
```

---

## Task 12: Forecast Experiment Dict Batch Consumption

**Files:**
- Modify: `torch_timeseries/experiments/forecast.py`
- Test: `tests/experiments/test_forecast_batch_contract.py`

- [ ] **Step 1: Write tests for processing both tuple and dict batches**

Create `tests/experiments/test_forecast_batch_contract.py`:

```python
import torch

from torch_timeseries.experiments.forecast import unpack_forecast_batch


def test_unpack_forecast_batch_accepts_legacy_tuple():
    batch = (
        torch.zeros(2, 4, 3),
        torch.zeros(2, 2, 1),
        torch.ones(2, 4, 3),
        torch.ones(2, 2, 1),
        torch.zeros(2, 4, 4),
        torch.zeros(2, 2, 4),
    )

    unpacked = unpack_forecast_batch(batch)

    assert unpacked["x"].shape == (2, 4, 3)
    assert unpacked["y"].shape == (2, 2, 1)
    assert unpacked["raw_y"].shape == (2, 2, 1)


def test_unpack_forecast_batch_accepts_dict():
    batch = {
        "x": torch.zeros(2, 4, 3),
        "y": torch.zeros(2, 2, 1),
        "raw_y": torch.ones(2, 2, 1),
        "x_time": torch.zeros(2, 4, 4),
        "y_time": torch.zeros(2, 2, 4),
    }

    unpacked = unpack_forecast_batch(batch)

    assert unpacked["x"].shape == (2, 4, 3)
    assert unpacked["raw_y"].shape == (2, 2, 1)
```

- [ ] **Step 2: Run forecast batch tests and verify they fail**

Run:

```bash
python -m pytest tests/experiments/test_forecast_batch_contract.py -q
```

Expected: FAIL because `unpack_forecast_batch` does not exist.

- [ ] **Step 3: Add unpack helper**

Modify `torch_timeseries/experiments/forecast.py` near imports:

```python
def unpack_forecast_batch(batch):
    if isinstance(batch, dict):
        return {
            "x": batch["x"],
            "y": batch["y"],
            "raw_x": batch.get("raw_x"),
            "raw_y": batch.get("raw_y"),
            "x_time": batch.get("x_time"),
            "y_time": batch.get("y_time"),
        }

    batch_x, batch_y, batch_origin_x, batch_origin_y, batch_x_date_enc, batch_y_date_enc = batch[:6]
    return {
        "x": batch_x,
        "y": batch_y,
        "raw_x": batch_origin_x,
        "raw_y": batch_origin_y,
        "x_time": batch_x_date_enc,
        "y_time": batch_y_date_enc,
    }
```

- [ ] **Step 4: Update `_train` and `_evaluate` loops to use helper**

In `_train`, replace tuple unpacking:

```python
for i, batch in enumerate(self.train_loader):
    unpacked = unpack_forecast_batch(batch)
    batch_x = unpacked["x"]
    batch_y = unpacked["y"]
    origin_x = unpacked["raw_x"]
    origin_y = unpacked["raw_y"]
    batch_x_date_enc = unpacked["x_time"]
    batch_y_date_enc = unpacked["y_time"]
```

In `_evaluate`, apply the same pattern.

When `self.invtrans_loss` is true, require `origin_y`:

```python
if self.invtrans_loss and origin_y is None:
    raise RuntimeError("invtrans_loss=True requires raw_y in the forecast batch")
```

- [ ] **Step 5: Run forecast batch tests**

Run:

```bash
python -m pytest tests/experiments/test_forecast_batch_contract.py -q
```

Expected: PASS.

- [ ] **Step 6: Run existing experiment runtime tests**

Run:

```bash
python -m pytest tests/experiments/test_autoformer.py tests/experiments/test_experiment_runtime_edges.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add torch_timeseries/experiments/forecast.py tests/experiments/test_forecast_batch_contract.py
git commit -m "refactor: support dict forecast batches"
```

---

## Task 13: Imputation Dict Batch Consumption

**Files:**
- Modify: `torch_timeseries/dataloader/maskts.py`
- Modify: `torch_timeseries/experiments/imputation.py`
- Test: `tests/experiments/test_imputation_batch_contract.py`

- [ ] **Step 1: Write tests for imputation batch unpacking**

Create `tests/experiments/test_imputation_batch_contract.py`:

```python
import torch

from torch_timeseries.experiments.imputation import unpack_imputation_batch


def test_unpack_imputation_batch_accepts_legacy_tuple():
    batch = (
        torch.zeros(2, 4, 3),
        torch.ones(2, 4, 3),
        torch.full((2, 4, 3), 2.0),
        torch.ones(2, 4, 3),
        torch.zeros(2, 4, 4),
    )

    unpacked = unpack_imputation_batch(batch)

    assert unpacked["x"].shape == (2, 4, 3)
    assert unpacked["masked_x"].shape == (2, 4, 3)
    assert unpacked["raw_x"].shape == (2, 4, 3)


def test_unpack_imputation_batch_accepts_dict():
    batch = {
        "masked_x": torch.zeros(2, 4, 3),
        "x": torch.ones(2, 4, 3),
        "raw_x": torch.full((2, 4, 3), 2.0),
        "mask": torch.ones(2, 4, 3),
        "x_time": torch.zeros(2, 4, 4),
    }

    unpacked = unpack_imputation_batch(batch)

    assert unpacked["x"].shape == (2, 4, 3)
    assert unpacked["x_time"].shape == (2, 4, 4)
```

- [ ] **Step 2: Run imputation tests and verify they fail**

Run:

```bash
python -m pytest tests/experiments/test_imputation_batch_contract.py -q
```

Expected: FAIL because `unpack_imputation_batch` does not exist.

- [ ] **Step 3: Add unpack helper**

Modify `torch_timeseries/experiments/imputation.py` near imports:

```python
def unpack_imputation_batch(batch):
    if isinstance(batch, dict):
        return {
            "masked_x": batch["masked_x"],
            "x": batch["x"],
            "raw_x": batch.get("raw_x"),
            "mask": batch["mask"],
            "x_time": batch.get("x_time"),
        }

    batch_masked_x, batch_x, batch_origin_x, batch_mask, batch_x_date_enc = batch
    return {
        "masked_x": batch_masked_x,
        "x": batch_x,
        "raw_x": batch_origin_x,
        "mask": batch_mask,
        "x_time": batch_x_date_enc,
    }
```

- [ ] **Step 4: Update imputation train/evaluate loops to use helper**

Replace tuple unpacking in `_train` and `_evaluate` with:

```python
for batch in dataloader:
    unpacked = unpack_imputation_batch(batch)
    batch_masked_x = unpacked["masked_x"]
    batch_x = unpacked["x"]
    batch_origin_x = unpacked["raw_x"]
    batch_mask = unpacked["mask"]
    batch_x_date_enc = unpacked["x_time"]
```

When `self.invtrans_loss` is true, require `batch_origin_x`.

- [ ] **Step 5: Run imputation batch tests**

Run:

```bash
python -m pytest tests/experiments/test_imputation_batch_contract.py -q
```

Expected: PASS.

- [ ] **Step 6: Run experiment tests**

Run:

```bash
python -m pytest tests/experiments/test_experiment_runtime_edges.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add torch_timeseries/experiments/imputation.py tests/experiments/test_imputation_batch_contract.py
git commit -m "refactor: support dict imputation batches"
```

---

## Task 14: Public Exports And Documentation

**Files:**
- Modify: `torch_timeseries/dataloader/__init__.py`
- Create: `docs/superpowers/specs/2026-05-27-timeseries-dataloader-v2-design.md`
- Modify: `README.md` or `docs/concepts/dataloader.rst`
- Test: `tests/dataloader/v2/test_public_exports.py`

- [ ] **Step 1: Write public export test**

Create `tests/dataloader/v2/test_public_exports.py`:

```python
from torch_timeseries.dataloader.v2 import (
    RatioSplit,
    TimeSeriesDataModule,
    WindowSpec,
    resolve_feature_selection,
)


def test_v2_public_exports_importable():
    assert RatioSplit is not None
    assert TimeSeriesDataModule is not None
    assert WindowSpec is not None
    assert resolve_feature_selection is not None
```

- [ ] **Step 2: Run export test**

Run:

```bash
python -m pytest tests/dataloader/v2/test_public_exports.py -q
```

Expected: PASS if prior tasks exported correctly.

- [ ] **Step 3: Export v2 module from dataloader package**

Modify `torch_timeseries/dataloader/__init__.py`:

```python
from . import v2
```

Add `"v2"` to `__all__`.

- [ ] **Step 4: Write design documentation**

Create `docs/superpowers/specs/2026-05-27-timeseries-dataloader-v2-design.md`:

```markdown
# Time Series Dataloader V2 Design

## Goals

- Separate raw dataset loading, split policy, scaling, time feature generation, windowing, and DataLoader construction.
- Support M->1, M->M, and M->K forecasting through `input_features` and `target_features`.
- Use `None` as the default selector meaning all features.
- Support datasets without column names by generating `feature_0`, `feature_1`, ...
- Return PyTorch-native batch dictionaries.
- Keep raw values in stable optional fields: `raw_x`, `raw_y`.
- Fit scalers only on train split.
- Decouple time feature dimensions from dataset `freq`.

## Batch Contract

Forecast batches use:

```python
{
    "x": Tensor,        # [B, context_length, len(input_features)]
    "y": Tensor,        # [B, prediction_length, len(target_features)]
    "x_time": Tensor,   # optional
    "y_time": Tensor,   # optional
    "raw_x": Tensor,    # optional
    "raw_y": Tensor,    # optional
    "x_index": Tensor,  # optional
    "y_index": Tensor,  # optional
}
```

Optional fields are omitted when disabled.

## Feature Selection

`input_features=None` and `target_features=None` mean all features. Selectors may be ints, negative ints, strings, or sequences of ints/strings.

## Scaling

`ScalerBundle` owns separate input and target scalers. It fits on train data only and allows target inverse transform for M->1 and M->K predictions without shape mismatch.

## Frequency And Time Features

Dataset `freq` remains sampling metadata. `TimeFeatureEncoder` decides generated features and exposes `output_dim`; models should receive `time_feature_dim` instead of inferring it from `freq`.

## Migration

Legacy tuple loaders are kept through adapters. New code should consume dict batches directly.
```

- [ ] **Step 5: Add short dataloader docs**

Modify `docs/concepts/dataloader.rst` by adding a section:

```rst
V2 Forecast Data Path
---------------------

The v2 data path separates feature selection, split strategies, scaling,
time feature generation, and windowing. ``input_features=None`` and
``target_features=None`` select all variables. Use an explicit selector
such as ``target_features=[-1]`` for multivariate-input single-target
forecasting.

V2 datasets return dictionaries rather than positional tuples. Forecast
batches contain ``x`` and ``y`` plus optional ``x_time``, ``y_time``,
``raw_x``, ``raw_y``, ``x_index``, and ``y_index`` fields.
```

- [ ] **Step 6: Run export and v2 tests**

Run:

```bash
python -m pytest tests/dataloader/v2 -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add torch_timeseries/dataloader/__init__.py docs/superpowers/specs/2026-05-27-timeseries-dataloader-v2-design.md docs/concepts/dataloader.rst tests/dataloader/v2/test_public_exports.py
git commit -m "docs: document dataloader v2 contract"
```

---

## Task 15: Full Verification And Cleanup

**Files:**
- No planned source files unless verification reveals issues.

- [ ] **Step 1: Run the full test suite**

Run:

```bash
python -m pytest -q
```

Expected: PASS. The local CUDA driver warning may appear but must not fail the run.

- [ ] **Step 2: Check formatting-sensitive whitespace**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 3: Check imports for public modules**

Run:

```bash
python - <<'PY'
from torch_timeseries.dataloader.v2 import TimeSeriesDataModule, WindowSpec, RatioSplit
from torch_timeseries.core import TimeSeriesDataset
from torch_timeseries.nn.embedding import TimeFeatureEmbedding
print(TimeSeriesDataModule, WindowSpec, RatioSplit, TimeSeriesDataset, TimeFeatureEmbedding)
PY
```

Expected: prints class objects and exits 0.

- [ ] **Step 4: Review compatibility risk**

Run:

```bash
python -m pytest tests/dataloader/slidingwindowts.py tests/experiments/test_autoformer.py tests/experiments/test_experiment_runtime_edges.py -q
```

Expected: PASS. This confirms old tuple-based experiment paths still work.

- [ ] **Step 5: Commit final verification-only fixes if needed**

If Step 1-4 required fixes, commit them:

```bash
git add <fixed-files>
git commit -m "fix: stabilize dataloader v2 migration"
```

If no fixes were required, do not create an empty commit.

---

## Self-Review

- Spec coverage: The plan covers `SlidingWindowTS`/ETT rationalization, raw/scaled batch policy, scaler placement, custom datasets, M->1/M->M/M->K feature selection, anonymous feature names, `freq`/`timeF` decoupling, and backward-compatible migration.
- Placeholder scan: No `TBD`, `TODO`, or unspecified implementation steps remain. Each code task includes concrete tests, implementation snippets, commands, and expected results.
- Type consistency: The plan consistently uses `FeatureSelector`, `FeatureSelection`, `WindowSpec`, `WindowedDataset`, `ScalerBundle`, `TimeFeatureEncoder`, `TimeSeriesDataModule`, and dict batch keys `x`, `y`, `raw_x`, `raw_y`, `x_time`, `y_time`, `x_index`, `y_index`.
