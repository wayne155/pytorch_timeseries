# time_enc String Aliases Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow `time_enc="fourier"` (and `"calendar"`, `"normalized"`) as readable string aliases in all v2 dataloader configs, while staying backward-compatible with existing `int` and `TimeEncoding` usage.

**Architecture:** Extend the existing `TimeEncoding` IntEnum with a `_missing_` hook that resolves case-insensitive string names; update type annotations in three v2 config/dataset classes to `Union[TimeEncoding, str, int]`; normalize strings to `TimeEncoding` at the boundary where values are passed to `time_features()`.

**Tech Stack:** Python 3.8+, `enum.IntEnum`, `dataclasses`, `torch.utils.data`

**Spec:** `docs/superpowers/specs/2026-06-08-time-enc-string-aliases-design.md`

---

## File Map

| File | Change |
|---|---|
| `torch_timeseries/utils/timefeatures.py` | Add `_missing_` to `TimeEncoding` |
| `torch_timeseries/dataloader/v2/forecast.py` | Type annotation + import + normalization in `_build_datasets` |
| `torch_timeseries/dataloader/v2/imputation.py` | Type annotation + import + normalization in `ImputationWindowedDataset.__init__` |
| `torch_timeseries/dataloader/v2/irregular_classification.py` | Type annotation + import only |
| `tests/test_cli_and_registry.py` | Add string-alias tests for `TimeEncoding._missing_` |
| `tests/dataloader/test_v2_task_modules.py` | Add string-alias integration tests for all three v2 configs |

---

## Task 1: Add `_missing_` to `TimeEncoding`

**Files:**
- Modify: `torch_timeseries/utils/timefeatures.py:11-15`
- Test: `tests/test_cli_and_registry.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli_and_registry.py` after the existing `test_time_encoding_enum_values_match_int_behaviour` function:

```python
def test_time_encoding_from_string_lowercase():
    from torch_timeseries.utils.timefeatures import TimeEncoding
    assert TimeEncoding("calendar") is TimeEncoding.CALENDAR
    assert TimeEncoding("fourier") is TimeEncoding.FOURIER
    assert TimeEncoding("normalized") is TimeEncoding.NORMALIZED


def test_time_encoding_from_string_uppercase():
    from torch_timeseries.utils.timefeatures import TimeEncoding
    assert TimeEncoding("FOURIER") is TimeEncoding.FOURIER
    assert TimeEncoding("Calendar") is TimeEncoding.CALENDAR


def test_time_encoding_invalid_string_raises():
    from torch_timeseries.utils.timefeatures import TimeEncoding
    import pytest
    with pytest.raises(ValueError):
        TimeEncoding("invalid")
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
pytest tests/test_cli_and_registry.py::test_time_encoding_from_string_lowercase tests/test_cli_and_registry.py::test_time_encoding_from_string_uppercase tests/test_cli_and_registry.py::test_time_encoding_invalid_string_raises -v
```

Expected: all three FAIL with `ValueError` (the enum rejects strings before `_missing_` exists).

- [ ] **Step 3: Add `_missing_` to `TimeEncoding`**

In `torch_timeseries/utils/timefeatures.py`, replace the `TimeEncoding` class:

```python
class TimeEncoding(IntEnum):
    CALENDAR   = 0   # raw integer calendar fields (year, month, day, ...)
    FOURIER    = 1   # normalized Fourier-style date features
    NORMALIZED = 3   # normalized calendar scalars

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            try:
                return cls[value.upper()]
            except KeyError:
                pass
        return None
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
pytest tests/test_cli_and_registry.py::test_time_encoding_from_string_lowercase tests/test_cli_and_registry.py::test_time_encoding_from_string_uppercase tests/test_cli_and_registry.py::test_time_encoding_invalid_string_raises tests/test_cli_and_registry.py::test_time_encoding_enum_values_match_int_behaviour -v
```

Expected: all four PASS.

- [ ] **Step 5: Commit**

```bash
git add torch_timeseries/utils/timefeatures.py tests/test_cli_and_registry.py
git commit -m "feat: add _missing_ string lookup to TimeEncoding"
```

---

## Task 2: Update `WindowConfig` in `forecast.py`

**Files:**
- Modify: `torch_timeseries/dataloader/v2/forecast.py`
- Test: `tests/dataloader/test_v2_task_modules.py`

The `_build_datasets` method (line ~120) builds a `common` dict and passes `time_enc=wc.time_enc` to `WindowedDataset`. The normalization one-liner goes right before that dict is constructed.

- [ ] **Step 1: Write the failing tests**

Add to `tests/dataloader/test_v2_task_modules.py`. First extend the existing top-of-file import:

```python
from torch_timeseries.dataloader.v2 import SplitConfig, LoaderConfig, TSBatch, ForecastDataModule, WindowConfig
```

Then add these test functions:

```python
def test_window_config_accepts_string_time_enc():
    """WindowConfig should accept string aliases without error."""
    cfg = WindowConfig(time_enc="fourier")
    assert cfg.time_enc == "fourier"


def _toy_forecast_dm(time_enc="calendar"):
    return ForecastDataModule(
        dataset=_ToyTS(root="/tmp"),
        scaler=StandardScaler(),
        window=WindowConfig(window=24, horizon=4, time_enc=time_enc, freq="h"),
        split=SplitConfig(train=0.7, val=0.1, test=0.2),
        loader=LoaderConfig(batch_size=8, num_workers=0),
    )


def test_forecast_dm_string_time_enc_calendar():
    dm = _toy_forecast_dm(time_enc="calendar")
    batch = next(iter(dm.train_loader))
    assert batch.x_time is not None


def test_forecast_dm_string_time_enc_fourier():
    dm = _toy_forecast_dm(time_enc="fourier")
    batch = next(iter(dm.train_loader))
    assert batch.x_time is not None


def test_forecast_dm_int_time_enc_still_works():
    dm = _toy_forecast_dm(time_enc=1)
    batch = next(iter(dm.train_loader))
    assert batch.x_time is not None
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
pytest tests/dataloader/test_v2_task_modules.py::test_forecast_dm_string_time_enc_calendar tests/dataloader/test_v2_task_modules.py::test_forecast_dm_string_time_enc_fourier -v
```

Expected: FAIL — `time_features` receives a string and raises `TypeError` or similar.

- [ ] **Step 3: Update `forecast.py`**

**a. Imports** — change line 14:
```python
from typing import Optional, Union
```
Add after the existing `from .._split import resolve_split_ratios` import line:
```python
from torch_timeseries.utils.timefeatures import TimeEncoding
```

**b. `WindowConfig` field** — change line 36:
```python
    time_enc: Union[TimeEncoding, str, int] = "calendar"
```

**c. `ForecastDataModule._build_datasets`** — add the normalization one-liner as the first line of the method body (before `wc = self.window_cfg` is used to build `common`):

```python
    def _build_datasets(self) -> None:
        wc = self.window_cfg
        time_enc = TimeEncoding(wc.time_enc) if not isinstance(wc.time_enc, TimeEncoding) else wc.time_enc
        common = dict(
            scaler=self.scaler,
            window=wc.window,
            horizon=wc.horizon,
            steps=wc.steps,
            stride=wc.stride,
            include_raw=wc.include_raw,
            include_time=wc.include_time,
            include_index=wc.include_index,
            single_variate=wc.single_variate,
            time_enc=time_enc,
            freq=wc.freq,
            input_columns=wc.input_columns,
            target_columns=wc.target_columns,
        )
        self.train_dataset = WindowedDataset(self.train_subset, **common)
        self.val_dataset = WindowedDataset(self.val_subset, **common)
        self.test_dataset = WindowedDataset(self.test_subset, **common)
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
pytest tests/dataloader/test_v2_task_modules.py::test_window_config_accepts_string_time_enc tests/dataloader/test_v2_task_modules.py::test_forecast_dm_string_time_enc_calendar tests/dataloader/test_v2_task_modules.py::test_forecast_dm_string_time_enc_fourier tests/dataloader/test_v2_task_modules.py::test_forecast_dm_int_time_enc_still_works -v
```

Expected: all four PASS.

- [ ] **Step 5: Commit**

```bash
git add torch_timeseries/dataloader/v2/forecast.py tests/dataloader/test_v2_task_modules.py
git commit -m "feat: accept string time_enc aliases in WindowConfig"
```

---

## Task 3: Update `ImputationWindowConfig` and `ImputationWindowedDataset`

**Files:**
- Modify: `torch_timeseries/dataloader/v2/imputation.py`
- Test: `tests/dataloader/test_v2_task_modules.py`

The normalization goes inside `ImputationWindowedDataset.__init__` (line ~61) before the call to `time_features()`. This means both direct use of the dataset class and use via `ImputationDataModule` are covered.

- [ ] **Step 1: Write the failing tests**

Add to `tests/dataloader/test_v2_task_modules.py`:

```python
def test_imputation_config_accepts_string_time_enc():
    cfg = ImputationWindowConfig(time_enc="fourier")
    assert cfg.time_enc == "fourier"


def test_imputation_dm_string_time_enc_fourier():
    dm = _toy_imputation_dm(window=32, time_enc="fourier", freq="h")
    batch = next(iter(dm.train_loader))
    assert batch.x_time is not None


def test_imputation_dm_string_time_enc_calendar():
    dm = _toy_imputation_dm(window=32, time_enc="calendar", freq="h")
    batch = next(iter(dm.train_loader))
    assert batch.x_time is not None


def test_imputation_dm_int_time_enc_still_works():
    dm = _toy_imputation_dm(window=32, time_enc=1, freq="h")
    batch = next(iter(dm.train_loader))
    assert batch.x_time is not None
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
pytest tests/dataloader/test_v2_task_modules.py::test_imputation_dm_string_time_enc_fourier tests/dataloader/test_v2_task_modules.py::test_imputation_dm_string_time_enc_calendar -v
```

Expected: FAIL.

- [ ] **Step 3: Update `imputation.py`**

**a. Imports** — change line 6:
```python
from typing import Optional, Union
```
Change line 12:
```python
from torch_timeseries.utils.timefeatures import time_features, TimeEncoding
```

**b. `ImputationWindowConfig` field** — change line 25:
```python
    time_enc: Union[TimeEncoding, str, int] = "calendar"
```

**c. `ImputationWindowedDataset.__init__` param** — change line 50:
```python
        time_enc: Union[TimeEncoding, str, int] = "calendar",
```
Add normalization on line 61, replacing the existing `time_features` call:
```python
        time_enc = TimeEncoding(time_enc) if not isinstance(time_enc, TimeEncoding) else time_enc
        self.time = time_features(subset.dates, time_enc, freq).astype(np.float32)
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
pytest tests/dataloader/test_v2_task_modules.py::test_imputation_config_accepts_string_time_enc tests/dataloader/test_v2_task_modules.py::test_imputation_dm_string_time_enc_fourier tests/dataloader/test_v2_task_modules.py::test_imputation_dm_string_time_enc_calendar tests/dataloader/test_v2_task_modules.py::test_imputation_dm_int_time_enc_still_works -v
```

Expected: all four PASS.

- [ ] **Step 5: Commit**

```bash
git add torch_timeseries/dataloader/v2/imputation.py tests/dataloader/test_v2_task_modules.py
git commit -m "feat: accept string time_enc aliases in ImputationWindowConfig"
```

---

## Task 4: Update `IrregularClassificationConfig`

**Files:**
- Modify: `torch_timeseries/dataloader/v2/irregular_classification.py`
- Test: `tests/dataloader/test_v2_task_modules.py`

Note: `time_enc` is not currently wired to `_IrregularClassificationDataset` (it normalises time to [0,1] per-sample instead). This task updates the type annotation for consistency; no normalization call is needed.

- [ ] **Step 1: Write the failing test**

Add to `tests/dataloader/test_v2_task_modules.py`. First add the import at the top alongside existing imports:

```python
from torch_timeseries.dataloader.v2.irregular_classification import IrregularClassificationConfig
```

Then add:

```python
def test_irregular_classification_config_accepts_string_time_enc():
    """IrregularClassificationConfig should accept string aliases without error."""
    from torch_timeseries.utils.timefeatures import TimeEncoding
    cfg = IrregularClassificationConfig(time_enc="fourier")
    assert cfg.time_enc == "fourier"

    cfg2 = IrregularClassificationConfig(time_enc=0)
    assert cfg2.time_enc == 0
```

- [ ] **Step 2: Run the test to confirm it passes already or fails as expected**

```bash
pytest tests/dataloader/test_v2_task_modules.py::test_irregular_classification_config_accepts_string_time_enc -v
```

Since `IrregularClassificationConfig` is a plain dataclass with `time_enc: int = 0`, assigning `"fourier"` will succeed at runtime (Python doesn't enforce dataclass type hints). The test should PASS already — confirm this. If it does, proceed to the implementation step to update the annotation for correctness; if it FAILs for some other reason, investigate before continuing.

- [ ] **Step 3: Update `irregular_classification.py`**

**a. Imports** — change line 5:
```python
from typing import List, Optional, Union
```
Add after line 13 (`from .irregular_batch import IrregularTSBatch, collate_irregular`):
```python
from torch_timeseries.utils.timefeatures import TimeEncoding
```

**b. `IrregularClassificationConfig` field** — change line 19:
```python
    time_enc: Union[TimeEncoding, str, int] = "calendar"
```
Remove the inline comment (it described the old int-only semantics):
```python
    freq: Optional[str] = None
```

- [ ] **Step 4: Run the full test suite for all changed files**

```bash
pytest tests/dataloader/test_v2_task_modules.py tests/test_cli_and_registry.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add torch_timeseries/dataloader/v2/irregular_classification.py tests/dataloader/test_v2_task_modules.py
git commit -m "feat: accept string time_enc aliases in IrregularClassificationConfig"
```

---

## Task 5: Full regression check

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: no regressions. All pre-existing tests pass.

- [ ] **Step 2: Spot-check the new API in a REPL**

```python
from torch_timeseries.dataloader.v2 import WindowConfig
from torch_timeseries.utils.timefeatures import TimeEncoding

# All of these should work without error
WindowConfig(time_enc="fourier")
WindowConfig(time_enc="CALENDAR")
WindowConfig(time_enc=TimeEncoding.NORMALIZED)
WindowConfig(time_enc=1)

# This should raise ValueError
try:
    WindowConfig(time_enc="bad")
    from torch_timeseries.dataloader.v2.forecast import ForecastDataModule
    # trigger normalization
except ValueError as e:
    print("Correctly raised:", e)
```
