# TODO Three Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address the three items in TODO.md: (1) multi-feature-to-multi-feature selection via `input_columns`/`target_columns`, (2) replace the ETT special-case branching in `ForecastExp` with the unified v2 `ForecastDataModule`, (3) rename magic `timeenc` integers to a `TimeEncoding` enum.

**Architecture:** Three independent improvements layered bottom-up: enum first (no behavior change), then column-selection in `WindowedDataset`, then wire both into `ForecastExp` by replacing the ETT/SlidingWindowTS branch with `ForecastDataModule`. Each task is self-contained and testable before the next begins.

**Tech Stack:** Python 3.8+, PyTorch, pandas, dataclasses, existing `torch_timeseries` package structure.

---

## File Map

| File | Change |
|---|---|
| `torch_timeseries/utils/timefeatures.py` | Add `TimeEncoding` enum; update `time_features()` signature to accept it; keep int fallback |
| `torch_timeseries/dataloader/v2/windowed.py` | Accept `TimeEncoding`; add `input_columns`/`target_columns` params |
| `torch_timeseries/dataloader/v2/forecast.py` | Add `input_columns`/`target_columns` to `WindowConfig`; pass `TimeEncoding` |
| `torch_timeseries/dataloader/v2/__init__.py` | Re-export `TimeEncoding` |
| `torch_timeseries/experiments/forecast.py` | Add `input_columns`/`target_columns` fields; replace ETT branching with `ForecastDataModule` |
| `tests/dataloader/test_split_and_uea.py` | New test: column selection shapes |
| `tests/test_cli_and_registry.py` | New test: `TimeEncoding` round-trip |

---

## Task 1: Add `TimeEncoding` enum

**Files:**
- Modify: `torch_timeseries/utils/timefeatures.py`

### Background

`time_features(dates, timeenc=0, freq='h')` currently switches on magic integers:
- `0` → raw calendar scalars (year, month, day, weekday, hour, minute, second)
- `1` → normalized Fourier-style date features
- `3` → normalized calendar scalars

These are passed through every dataloader and experiment as a bare `int`, making the call sites unreadable.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli_and_registry.py`:

```python
def test_time_encoding_enum_values_match_int_behaviour():
    import numpy as np
    import pandas as pd
    from torch_timeseries.utils.timefeatures import time_features, TimeEncoding

    dates = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=5, freq="h")})

    out_int  = time_features(dates, timeenc=0, freq="h")
    out_enum = time_features(dates, timeenc=TimeEncoding.CALENDAR, freq="h")
    np.testing.assert_array_equal(out_int, out_enum)

    out_int1  = time_features(dates, timeenc=1, freq="h")
    out_enum1 = time_features(dates, timeenc=TimeEncoding.FOURIER, freq="h")
    np.testing.assert_array_equal(out_int1, out_enum1)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /data/yww/notebook/pytorchtimseries
python -m pytest tests/test_cli_and_registry.py::test_time_encoding_enum_values_match_int_behaviour -v
```

Expected: `ImportError: cannot import name 'TimeEncoding'`

- [ ] **Step 3: Add `TimeEncoding` enum and update `time_features`**

In `torch_timeseries/utils/timefeatures.py`, add just below the imports at the top of the file:

```python
from enum import IntEnum

class TimeEncoding(IntEnum):
    CALENDAR   = 0   # raw integer calendar fields (year, month, day, ...)
    FOURIER    = 1   # normalized Fourier-style date features
    NORMALIZED = 3   # normalized calendar scalars
```

Change the `time_features` signature from:

```python
def time_features(dates:pd.DataFrame, timeenc=0, freq='h') -> np.ndarray:
```

to:

```python
def time_features(dates: pd.DataFrame, timeenc: "int | TimeEncoding" = TimeEncoding.CALENDAR, freq: str = "h") -> np.ndarray:
```

Because `TimeEncoding` is an `IntEnum`, the existing `if timeenc==0:` / `elif timeenc==1:` / `elif timeenc==3:` branches work unchanged — no other edits needed in this file.

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_cli_and_registry.py::test_time_encoding_enum_values_match_int_behaviour -v
```

Expected: `PASSED`

- [ ] **Step 5: Re-export from v2 package**

In `torch_timeseries/dataloader/v2/__init__.py`, add:

```python
from torch_timeseries.utils.timefeatures import TimeEncoding
```

And add `"TimeEncoding"` to `__all__`.

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest tests/ -q
```

Expected: all green, 0 failures.

- [ ] **Step 7: Commit**

```bash
git add torch_timeseries/utils/timefeatures.py torch_timeseries/dataloader/v2/__init__.py tests/test_cli_and_registry.py
git commit -m "feat: add TimeEncoding enum for time_features; IntEnum keeps int compat"
```

---

## Task 2: `input_columns` / `target_columns` in `WindowedDataset` and `WindowConfig`

**Files:**
- Modify: `torch_timeseries/dataloader/v2/windowed.py`
- Modify: `torch_timeseries/dataloader/v2/forecast.py`
- Modify: `torch_timeseries/dataloader/v2/__init__.py`
- Test: `tests/dataloader/test_split_and_uea.py`

### Background

`WindowedDataset.__getitem__` currently returns `x` and `y` that always span **all** features.
The TODO asks for independent input and target column selection:
- `input_columns=[0,1,2]` → `x` has shape `(window, 3)`
- `target_columns=[6]`    → `y` has shape `(steps, 1)`
- When both are `None` (default), all-features behaviour is unchanged.

`WindowConfig` is the public API surface; it needs two new optional list fields.

- [ ] **Step 1: Write the failing test**

Add to `tests/dataloader/test_split_and_uea.py`:

```python
def test_windowed_dataset_input_output_column_selection():
    import numpy as np
    import pandas as pd
    from torch_timeseries.core import TimeSeriesDataset, Freq, TimeseriesSubset
    from torch_timeseries.scaler import StandardScaler
    from torch_timeseries.dataloader.v2.windowed import WindowedDataset

    class _Toy(TimeSeriesDataset):
        name = "toy"; num_features = 5; freq = Freq.hours
        def download(self): pass
        def _load(self):
            n = 300
            rng = np.random.default_rng(42)
            self.df = pd.DataFrame(
                {"date": pd.date_range("2020-01-01", periods=n, freq="h"),
                 **{f"c{i}": rng.normal(size=n) for i in range(5)}}
            )
            self.dates = self.df[["date"]]
            self.data  = self.df.drop("date", axis=1).values
            self.length = n

    ds     = _Toy()
    subset = TimeseriesSubset(ds, range(len(ds)))
    scaler = StandardScaler()
    scaler.fit(subset.data)

    wd = WindowedDataset(
        subset, scaler,
        window=24, horizon=1, steps=12,
        input_columns=[0, 1, 2],
        target_columns=[4],
        include_time=False, include_raw=False,
    )
    batch = wd[0]
    assert batch.x.shape == (24, 3), batch.x.shape
    assert batch.y.shape == (12, 1), batch.y.shape
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/dataloader/test_split_and_uea.py::test_windowed_dataset_input_output_column_selection -v
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'input_columns'`

- [ ] **Step 3: Add columns params to `WindowedDataset`**

In `torch_timeseries/dataloader/v2/windowed.py`:

Add to `__init__` signature (after `freq`):

```python
    input_columns: Optional[list] = None,
    target_columns: Optional[list] = None,
```

Store them:

```python
        self.input_columns  = list(input_columns)  if input_columns  is not None else None
        self.target_columns = list(target_columns) if target_columns is not None else None
```

Update `num_features` (used externally):

```python
        if input_columns is not None:
            self.num_features = len(input_columns)
```

In `__getitem__`, after the lines that compute `x` and `y` from `self._scaled`:

```python
        x = self._scaled[x_slice]
        y = self._scaled[y_slice]
```

Add column slicing immediately after:

```python
        if self.input_columns is not None:
            x = x[:, self.input_columns]
        if self.target_columns is not None:
            y = y[:, self.target_columns]
```

Apply the same slicing to `x_raw`/`y_raw` inside `_slice`. Replace the existing `_slice` helper with:

```python
        def _slice(arr, x_cols=None, y_cols=None):
            if arr is None:
                return None, None
            xv, yv = arr[x_slice], arr[y_slice]
            if ch is not None and arr.ndim == 2:
                xv, yv = xv[:, ch:ch + 1], yv[:, ch:ch + 1]
            else:
                if x_cols is not None:
                    xv = xv[:, x_cols]
                if y_cols is not None:
                    yv = yv[:, y_cols]
            return xv, yv

        x_raw, y_raw = _slice(self._raw,  self.input_columns, self.target_columns)
        x_time, y_time = _slice(self._time)
        x_index, y_index = _slice(self._index)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/dataloader/test_split_and_uea.py::test_windowed_dataset_input_output_column_selection -v
```

Expected: `PASSED`

- [ ] **Step 5: Add `input_columns`/`target_columns` to `WindowConfig`**

In `torch_timeseries/dataloader/v2/forecast.py`, update `WindowConfig`:

```python
@dataclass
class WindowConfig:
    window: int = 168
    horizon: int = 1
    steps: int = 1
    stride: int = 1
    include_raw: bool = True
    include_time: bool = True
    include_index: bool = False
    single_variate: bool = False
    time_enc: "int | TimeEncoding" = 0
    freq: Optional[str] = None
    input_columns: Optional[list] = None
    target_columns: Optional[list] = None
```

In `ForecastDataModule._build_datasets`, pass these through to `WindowedDataset`:

```python
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
            time_enc=wc.time_enc,
            freq=wc.freq,
            input_columns=wc.input_columns,
            target_columns=wc.target_columns,
        )
```

Add `num_target_features` property to `ForecastDataModule`:

```python
    @property
    def num_target_features(self) -> int:
        wc = self.window_cfg
        if wc.target_columns is not None:
            return len(wc.target_columns)
        return self.dataset.num_features
```

Update `torch_timeseries/dataloader/v2/__init__.py` — `WindowConfig` is already exported, nothing more needed.

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest tests/ -q
python examples/v2_forecast_smoke.py
```

Expected: all green, `ALL OK`.

- [ ] **Step 7: Commit**

```bash
git add torch_timeseries/dataloader/v2/windowed.py torch_timeseries/dataloader/v2/forecast.py tests/dataloader/test_split_and_uea.py
git commit -m "feat: add input_columns/target_columns to WindowedDataset and WindowConfig"
```

---

## Task 3: Replace ETT special-casing in `ForecastExp` with `ForecastDataModule`

**Files:**
- Modify: `torch_timeseries/experiments/forecast.py`
- Test: `tests/experiments/test_experiment_runtime_edges.py`

### Background

`ForecastExp._init_data_loader` has three branches:

```python
if self.dataset_type[0:3] == "ETT":
    if self.dataset_type[0:4] == "ETTh":
        self.dataloader = ETTHLoader(...)
    elif self.dataset_type[0:4] == "ETTm":
        self.dataloader = ETTMLoader(...)
else:
    self.dataloader = SlidingWindowTS(...)
```

`ETTHLoader` and `ETTMLoader` are wrappers around `SlidingWindowTS` with ETT-specific defaults and have no behavioural difference from the general case. `ForecastDataModule` (v2) handles all datasets uniformly.

Also, `timeenc` is currently hardcoded to `1` via:
```python
embed = 'timeF'
timeenc = 0 if embed != 'timeF' else 1
```
This should become a proper `ForecastSettings` field.

- [ ] **Step 1: Write the failing test**

Add to `tests/experiments/test_experiment_runtime_edges.py`:

```python
def test_forecast_exp_uses_forecast_data_module(monkeypatch):
    """ForecastExp._init_data_loader must build a ForecastDataModule, not ETTHLoader."""
    from torch_timeseries.experiments.forecast import ForecastExp
    from torch_timeseries.dataloader.v2 import ForecastDataModule

    built = {}

    _orig_init = ForecastDataModule.__init__

    def _spy_init(self, *args, **kwargs):
        built["called"] = True
        _orig_init(self, *args, **kwargs)

    monkeypatch.setattr(ForecastDataModule, "__init__", _spy_init)

    class FakeForecast(ForecastExp):
        model_type = "DLinear"
        dataset_type = "ETTh1"

        def _init_dataset(self):
            import numpy as np
            import pandas as pd
            from torch_timeseries.core import TimeSeriesDataset, Freq

            class _DS(TimeSeriesDataset):
                name = "ETTh1"; num_features = 7; freq = Freq.hours
                def download(self): pass
                def _load(self):
                    n = 500
                    rng = np.random.default_rng(0)
                    self.df = pd.DataFrame(
                        {"date": pd.date_range("2020-01-01", periods=n, freq="h"),
                         **{f"c{i}": rng.normal(size=n) for i in range(7)}}
                    )
                    self.dates = self.df[["date"]]
                    self.data  = self.df.drop("date", axis=1).values
                    self.length = n

            self.dataset = _DS(root="/tmp")

    exp = FakeForecast()
    exp._init_data_loader()

    assert built.get("called"), "ForecastDataModule was not constructed"
    assert hasattr(exp, "train_loader")
    assert hasattr(exp, "val_loader")
    assert hasattr(exp, "test_loader")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/experiments/test_experiment_runtime_edges.py::test_forecast_exp_uses_forecast_data_module -v
```

Expected: `AssertionError: ForecastDataModule was not constructed`

- [ ] **Step 3: Add `time_enc` field to `ForecastSettings` and refactor `_init_data_loader`**

In `torch_timeseries/experiments/forecast.py`:

Update the imports at the top to include:

```python
from ..dataloader.v2 import ForecastDataModule, WindowConfig, SplitConfig, LoaderConfig
from ..utils.timefeatures import TimeEncoding
```

Update `ForecastSettings` dataclass:

```python
@dataclass
class ForecastSettings:
    horizon: int = 1
    windows: int = 96
    pred_len: int = 96
    train_ratio: float = 0.7
    test_ratio: float = 0.2
    time_enc: int = 1          # 0=CALENDAR, 1=FOURIER, 3=NORMALIZED
    input_columns: List[int] = field(default_factory=list)
    target_columns: List[int] = field(default_factory=list)
```

Replace `_init_data_loader` entirely with:

```python
    def _init_data_loader(self):
        self._init_dataset()

        self.scaler = parse_type(self.scaler_type, globals=globals())()

        window_cfg = WindowConfig(
            window=self.windows,
            horizon=self.horizon,
            steps=self.pred_len,
            time_enc=self.time_enc,
            freq=self.dataset.freq,
            input_columns=self.input_columns or None,
            target_columns=self.target_columns or None,
        )
        split_cfg = SplitConfig(
            train=self.train_ratio,
            test=self.test_ratio,
        )
        loader_cfg = LoaderConfig(
            batch_size=self.batch_size,
            num_workers=self.num_worker,
            shuffle_train=True,
        )
        self.datamodule = ForecastDataModule(
            dataset=self.dataset,
            scaler=self.scaler,
            window=window_cfg,
            split=split_cfg,
            loader=loader_cfg,
        )
        self.train_loader = self.datamodule.train_loader
        self.val_loader   = self.datamodule.val_loader
        self.test_loader  = self.datamodule.test_loader
        self.train_steps  = len(self.train_loader.dataset)
        self.val_steps    = len(self.val_loader.dataset)
        self.test_steps   = len(self.test_loader.dataset)

        print(f"train steps: {self.train_steps}")
        print(f"val steps:   {self.val_steps}")
        print(f"test steps:  {self.test_steps}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/experiments/test_experiment_runtime_edges.py::test_forecast_exp_uses_forecast_data_module -v
```

Expected: `PASSED`

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -q
python examples/v2_forecast_smoke.py
```

Expected: all green, `ALL OK`.

- [ ] **Step 6: Commit**

```bash
git add torch_timeseries/experiments/forecast.py tests/experiments/test_experiment_runtime_edges.py
git commit -m "refactor: ForecastExp uses ForecastDataModule; drop ETT special-case branches; add time_enc/input_columns/target_columns fields"
```

---

## Self-Review

**Spec coverage:**
- [x] Feature selection, multiple features predict multiple features → Task 2 + Task 3 (`input_columns`/`target_columns` in `WindowConfig` and `ForecastSettings`)
- [x] Better architecture to various types of tasks → Task 3 (removes ETT branching, unifies under `ForecastDataModule`)
- [x] `timeenc` renamed → Task 1 (`TimeEncoding` enum) + Task 3 (`time_enc` field in settings)

**Placeholder scan:** None found — all steps contain exact code.

**Type consistency:**
- `input_columns: Optional[list]` used in `WindowConfig`, `WindowedDataset.__init__`, and `ForecastSettings` consistently
- `TimeEncoding` defined in Task 1 and used in Task 3 — both reference `torch_timeseries.utils.timefeatures.TimeEncoding`
- `ForecastDataModule` used in Task 3 is the same class tested in Task 2
