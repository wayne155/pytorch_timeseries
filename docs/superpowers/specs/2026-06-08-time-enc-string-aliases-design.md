# Design: `time_enc` String Aliases for v2 Dataloader Configs

**Date:** 2026-06-08  
**Scope:** v2 dataloader layer only  
**Status:** Approved

---

## Problem

`WindowConfig.time_enc` (and equivalent fields in the imputation and irregular-classification configs) is typed as `int` with magic values `0`, `1`, `3`. Users must consult source or docs to know what each integer means.

## Goal

Allow `time_enc="fourier"` (and `"calendar"`, `"normalized"`) as readable string aliases, while keeping existing `int` and `TimeEncoding` enum usage fully backward-compatible.

---

## Design

### 1. Extend `TimeEncoding` with `_missing_`

**File:** `torch_timeseries/utils/timefeatures.py`

Add a `_missing_` classmethod to the existing `TimeEncoding(IntEnum)`:

```python
@classmethod
def _missing_(cls, value):
    if isinstance(value, str):
        try:
            return cls[value.upper()]
        except KeyError:
            pass
    return None
```

Effect: `TimeEncoding("fourier")` → `TimeEncoding.FOURIER`, case-insensitive. Invalid strings raise `ValueError` automatically (Python's `IntEnum` behaviour when `_missing_` returns `None`).

Valid string aliases (case-insensitive):
| String | Resolves to | Int value |
|---|---|---|
| `"calendar"` | `TimeEncoding.CALENDAR` | `0` |
| `"fourier"` | `TimeEncoding.FOURIER` | `1` |
| `"normalized"` | `TimeEncoding.NORMALIZED` | `3` |

### 2. Update v2 config type annotations

**Files:**
- `torch_timeseries/dataloader/v2/forecast.py` — `WindowConfig`
- `torch_timeseries/dataloader/v2/imputation.py` — `ImputationWindowConfig` and `ImputationWindowedDataset.__init__` param
- `torch_timeseries/dataloader/v2/irregular_classification.py` — `IrregularClassificationConfig`

Change:
```python
time_enc: int = 0
```
To:
```python
time_enc: Union[TimeEncoding, str, int] = "calendar"
```

Add `from torch_timeseries.utils.timefeatures import TimeEncoding` to each file's imports.

Default changes from `0` to `"calendar"` — semantically identical, more readable.

### 3. Normalize at dataset-build time

In each module, before passing `time_enc` downstream to `WindowedDataset` / `time_features`, add:

```python
time_enc = TimeEncoding(time_enc) if not isinstance(time_enc, TimeEncoding) else time_enc
```

Locations:
- `ForecastDataModule._build_datasets()` — before passing `wc.time_enc` to `WindowedDataset`
- `ImputationDataModule._build_datasets()` — before passing `wc.time_enc` to `ImputationWindowedDataset`
- `ImputationWindowedDataset.__init__` — before passing `time_enc` to `time_features()`
- `IrregularClassificationConfig.time_enc` — type annotation only; `time_enc` is currently **not wired** to `_IrregularClassificationDataset` (which normalises time to [0,1] per-sample instead). No normalization call needed there until the wiring is added.

Since `TimeEncoding` is an `IntEnum`, the resolved value satisfies all existing `if timeenc == 0 / 1 / 3` checks in `time_features()` with no further changes.

---

## Backward Compatibility

| Existing usage | Still works? |
|---|---|
| `time_enc=0` | Yes — `TimeEncoding(0)` → `TimeEncoding.CALENDAR` |
| `time_enc=1` | Yes — `TimeEncoding(1)` → `TimeEncoding.FOURIER` |
| `time_enc=TimeEncoding.FOURIER` | Yes — already a `TimeEncoding`, no-op |
| `time_enc="fourier"` | Yes — new |

---

## Out of Scope

- v1 dataloaders (`sliding_window_ts.py`, `noverlap_window_ts.py`, etc.) — unchanged
- `time_features()` function internals — unchanged
- Any model code — unchanged

---

## Files Changed

1. `torch_timeseries/utils/timefeatures.py` — add `_missing_`
2. `torch_timeseries/dataloader/v2/forecast.py` — type annotation + normalization
3. `torch_timeseries/dataloader/v2/imputation.py` — type annotation + normalization
4. `torch_timeseries/dataloader/v2/irregular_classification.py` — type annotation + normalization
