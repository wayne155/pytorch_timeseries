# Foundation: Results Schema + DataModule Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a leaderboard-ready `RunResult` / `LocalBackend` results layer and three new v2 DataModules (`ImputationDataModule`, `AnomalyDataModule`, `UEADataModule`) that all return `TSBatch`.

**Architecture:** `RunResult` is a pure dataclass saved as JSON by `LocalBackend`. The three new DataModules mirror `ForecastDataModule` in interface (same `train_loader` / `val_loader` / `test_loader` properties, same `SplitConfig` / `LoaderConfig`). Each has its own task-specific window config class.

**Tech Stack:** Python dataclasses, PyTorch `Dataset`/`DataLoader`, existing `TSBatch` / `collate_tsbatch`, `resolve_split_ratios`, `seed_worker`, `time_features`, `TimeseriesSubset`.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `torch_timeseries/results/__init__.py` | Re-exports |
| Create | `torch_timeseries/results/schema.py` | `RunResult` dataclass |
| Create | `torch_timeseries/results/backends.py` | `ResultBackend`, `LocalBackend`, `WandbBackend`, `_get_git_commit` |
| Create | `torch_timeseries/dataloader/v2/imputation.py` | `ImputationWindowConfig`, `ImputationWindowedDataset`, `ImputationDataModule` |
| Create | `torch_timeseries/dataloader/v2/anomaly.py` | `AnomalyWindowConfig`, `AnomalyWindowedDataset`, `AnomalyDataModule` |
| Create | `torch_timeseries/dataloader/v2/uea.py` | `UEAWindowConfig`, `UEAWindowedDataset`, `UEADataModule` |
| Modify | `torch_timeseries/dataloader/v2/__init__.py` | Add new exports |
| Create | `tests/results/__init__.py` | (empty) |
| Create | `tests/results/test_backends.py` | Tests for `RunResult` + `LocalBackend` + `WandbBackend` |
| Create | `tests/dataloader/test_v2_task_modules.py` | Tests for three new DataModules |

---

### Task 1: RunResult dataclass + LocalBackend

**Files:**
- Create: `torch_timeseries/results/schema.py`
- Create: `torch_timeseries/results/backends.py`
- Create: `torch_timeseries/results/__init__.py`
- Create: `tests/results/__init__.py`
- Create: `tests/results/test_backends.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/results/test_backends.py
import json
import pytest
from torch_timeseries.results.schema import RunResult
from torch_timeseries.results.backends import LocalBackend


def _r(**kw):
    base = dict(
        model="DLinear", task="Forecast", dataset="ETTh1", seed=1,
        timestamp="2026-01-01T00:00:00",
        hparams={"lr": 0.001, "windows": 96},
        metrics={"mse": 0.382, "mae": 0.271},
        num_params=22000, train_time_sec=12.5, git_commit="abc123",
    )
    base.update(kw)
    return RunResult(**base)


def test_run_result_has_required_fields():
    r = _r()
    assert r.history is None          # reserved, defaults None
    assert isinstance(r.hparams, dict)
    assert isinstance(r.metrics, dict)


def test_local_backend_saves_json(tmp_path):
    backend = LocalBackend(save_dir=str(tmp_path))
    backend.save(_r())
    fname = tmp_path / "DLinear_Forecast_ETTh1_seed1.json"
    assert fname.exists()
    d = json.loads(fname.read_text())
    assert d["model"] == "DLinear"
    assert d["metrics"]["mse"] == pytest.approx(0.382)
    assert d["history"] is None


def test_local_backend_load_all_no_filter(tmp_path):
    backend = LocalBackend(save_dir=str(tmp_path))
    backend.save(_r(seed=1))
    backend.save(_r(seed=2))
    backend.save(_r(model="Autoformer", seed=1))
    assert len(backend.load_all()) == 3


def test_local_backend_load_all_with_filter(tmp_path):
    backend = LocalBackend(save_dir=str(tmp_path))
    backend.save(_r(seed=1))
    backend.save(_r(seed=2))
    backend.save(_r(model="Autoformer", seed=1))
    results = backend.load_all(model="DLinear")
    assert len(results) == 2
    assert all(r.model == "DLinear" for r in results)


def test_local_backend_overwrite(tmp_path):
    backend = LocalBackend(save_dir=str(tmp_path))
    backend.save(_r(metrics={"mse": 0.5}))
    backend.save(_r(metrics={"mse": 0.3}))          # same filename → overwrite
    results = backend.load_all()
    assert len(results) == 1
    assert results[0].metrics["mse"] == pytest.approx(0.3)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/results/test_backends.py -v
```

Expected: `ModuleNotFoundError: No module named 'torch_timeseries.results'`

- [ ] **Step 3: Create `torch_timeseries/results/schema.py`**

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class RunResult:
    model: str
    task: str
    dataset: str
    seed: int
    timestamp: str
    hparams: dict
    metrics: dict
    num_params: int
    train_time_sec: float
    git_commit: str
    history: Optional[dict] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        return cls(**d)
```

- [ ] **Step 4: Create `torch_timeseries/results/backends.py`**

```python
from __future__ import annotations

import json
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import List

from .schema import RunResult


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


class ResultBackend(ABC):
    @abstractmethod
    def save(self, result: RunResult) -> None: ...

    @abstractmethod
    def load_all(self, **filters) -> List[RunResult]: ...


class LocalBackend(ResultBackend):
    def __init__(self, save_dir: str = "./results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _filename(self, r: RunResult) -> str:
        return os.path.join(
            self.save_dir,
            f"{r.model}_{r.task}_{r.dataset}_seed{r.seed}.json",
        )

    def save(self, result: RunResult) -> None:
        with open(self._filename(result), "w") as f:
            json.dump(asdict(result), f, indent=2)

    def load_all(self, **filters) -> List[RunResult]:
        results = []
        for fname in os.listdir(self.save_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(self.save_dir, fname)) as f:
                d = json.load(f)
            r = RunResult.from_dict(d)
            if all(getattr(r, k, None) == v for k, v in filters.items()):
                results.append(r)
        return results


class WandbBackend(ResultBackend):
    """Save results to Weights & Biases. Requires ``pip install wandb``."""

    def __init__(self, project: str, entity: str = None):
        try:
            import wandb as _w
            self._wandb = _w
        except ImportError:
            raise ImportError(
                "wandb is required for WandbBackend: pip install wandb"
            )
        self.project = project
        self.entity = entity

    def save(self, result: RunResult) -> None:
        run = self._wandb.init(
            project=self.project,
            entity=self.entity,
            name=f"{result.model}_{result.task}_{result.dataset}_seed{result.seed}",
            config={**result.hparams, "model": result.model, "task": result.task,
                    "dataset": result.dataset, "seed": result.seed},
            reinit=True,
        )
        self._wandb.log(result.metrics)
        self._wandb.run.summary.update({
            "num_params": result.num_params,
            "train_time_sec": result.train_time_sec,
            "git_commit": result.git_commit,
        })
        run.finish()

    def load_all(self, **filters) -> List[RunResult]:
        # W&B is write-only for this backend; use LocalBackend for reads.
        return []
```

- [ ] **Step 5: Create `torch_timeseries/results/__init__.py`**

```python
from .schema import RunResult
from .backends import ResultBackend, LocalBackend, WandbBackend, _get_git_commit

__all__ = ["RunResult", "ResultBackend", "LocalBackend", "WandbBackend", "_get_git_commit"]
```

- [ ] **Step 6: Create `tests/results/__init__.py`** (empty file)

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest tests/results/test_backends.py -v
```

Expected: 5 PASSED

- [ ] **Step 8: Commit**

```bash
git add torch_timeseries/results/ tests/results/
git commit -m "feat: add RunResult dataclass and LocalBackend/WandbBackend"
```

---

### Task 2: ImputationDataModule

**Files:**
- Create: `torch_timeseries/dataloader/v2/imputation.py`
- Modify: `torch_timeseries/dataloader/v2/__init__.py`
- Create: `tests/dataloader/test_v2_task_modules.py`

**Background:** Imputation reconstructs a masked window. The DataModule returns `TSBatch(x=scaled_window, y=scaled_window)` — both `x` and `y` are copies of the same data. The mask (zeroing out random entries of `x`) is applied by the experiment's `_prepare_batch`, not here.

- [ ] **Step 1: Write the failing test**

```python
# tests/dataloader/test_v2_task_modules.py
import numpy as np
import pandas as pd
import pytest
import torch

from torch_timeseries.core import TimeSeriesDataset, Freq
from torch_timeseries.scaler import StandardScaler
from torch_timeseries.dataloader.v2 import SplitConfig, LoaderConfig, TSBatch
from torch_timeseries.dataloader.v2.imputation import (
    ImputationDataModule, ImputationWindowConfig,
)


class _ToyTS(TimeSeriesDataset):
    name = "toy_ts"
    num_features = 4
    freq = Freq.hours

    def download(self): pass

    def _load(self):
        n = 500
        rng = np.random.default_rng(0)
        self.df = pd.DataFrame(
            {"date": pd.date_range("2020-01-01", periods=n, freq="h"),
             **{f"c{i}": rng.normal(size=n) for i in range(4)}}
        )
        self.dates = self.df[["date"]]
        self.data = self.df.drop("date", axis=1).values
        self.length = n


def _toy_imputation_dm(window=32, **kw):
    return ImputationDataModule(
        dataset=_ToyTS(root="/tmp"),
        scaler=StandardScaler(),
        window=ImputationWindowConfig(window=window, **kw),
        split=SplitConfig(train=0.7, val=0.1, test=0.2),
        loader=LoaderConfig(batch_size=16, num_workers=0),
    )


def test_imputation_dm_returns_tsbatch():
    dm = _toy_imputation_dm()
    batch = next(iter(dm.train_loader))
    assert isinstance(batch, TSBatch)


def test_imputation_dm_batch_shapes():
    dm = _toy_imputation_dm(window=32)
    batch = next(iter(dm.train_loader))
    B = min(16, len(dm.train_dataset))
    assert batch.x.shape == (B, 32, 4)
    assert batch.y.shape == (B, 32, 4)


def test_imputation_dm_x_y_equal_values():
    """y must equal x before masking is applied (reconstruction target)."""
    dm = _toy_imputation_dm(window=32)
    batch = next(iter(dm.train_loader))
    assert torch.allclose(batch.x, batch.y)


def test_imputation_dm_x_y_independent_tensors():
    """x and y must be separate tensors so masking x doesn't corrupt y."""
    dm = _toy_imputation_dm(window=32)
    batch = next(iter(dm.train_loader))
    assert batch.x.data_ptr() != batch.y.data_ptr()


def test_imputation_dm_has_all_loaders():
    dm = _toy_imputation_dm()
    assert hasattr(dm, "train_loader")
    assert hasattr(dm, "val_loader")
    assert hasattr(dm, "test_loader")


def test_imputation_dm_mask_ratio_accessible():
    dm = _toy_imputation_dm(mask_ratio=0.3)
    assert dm.mask_ratio == pytest.approx(0.3)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/dataloader/test_v2_task_modules.py -v
```

Expected: `ModuleNotFoundError: No module named 'torch_timeseries.dataloader.v2.imputation'`

- [ ] **Step 3: Create `torch_timeseries/dataloader/v2/imputation.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from torch_timeseries.core import TimeSeriesDataset, TimeseriesSubset
from torch_timeseries.utils.timefeatures import time_features

from .._seed import seed_worker
from .._split import resolve_split_ratios
from .batch import TSBatch, collate_tsbatch
from .forecast import SplitConfig, LoaderConfig


@dataclass
class ImputationWindowConfig:
    window: int = 96
    mask_ratio: float = 0.25
    stride: int = 1
    time_enc: int = 0
    freq: Optional[str] = None


class ImputationWindowedDataset(Dataset):
    """Sliding-window dataset for masked reconstruction.

    Returns TSBatch where x == y (both are copies of the same scaled window).
    The calling experiment applies a random mask to x in _prepare_batch.
    """

    def __init__(
        self,
        subset: TimeseriesSubset,
        scaler,
        window: int = 96,
        stride: int = 1,
        time_enc: int = 0,
        freq: Optional[str] = None,
    ) -> None:
        self.window = window
        freq = freq or str(subset.freq)
        self.scaled = scaler.transform(subset.data).astype(np.float32)
        self.raw = subset.data.astype(np.float32)
        self.time = time_features(subset.dates, time_enc, freq).astype(np.float32)
        self._starts = list(range(0, len(subset) - window + 1, stride))

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> TSBatch:
        s = self._starts[idx]
        e = s + self.window
        x = torch.from_numpy(self.scaled[s:e].copy())
        return TSBatch(
            x=x,
            y=x.clone(),                                     # independent copy
            x_time=torch.from_numpy(self.time[s:e].copy()),
            x_raw=torch.from_numpy(self.raw[s:e].copy()),
        )


class ImputationDataModule:
    """Wires dataset -> scaler -> split -> ImputationWindowedDataset -> DataLoader."""

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        scaler,
        window: ImputationWindowConfig = None,
        split: SplitConfig = None,
        loader: LoaderConfig = None,
        scale_in_train: bool = True,
    ) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.window_cfg = window or ImputationWindowConfig()
        self.split_cfg = split or SplitConfig()
        self.loader_cfg = loader or LoaderConfig()
        self.scale_in_train = scale_in_train

        self._build_subsets()
        self._fit_scaler()
        self._build_datasets()
        self._build_loaders()

    def _build_subsets(self) -> None:
        train, val, test = resolve_split_ratios(
            train_ratio=self.split_cfg.train,
            val_ratio=self.split_cfg.val,
            test_ratio=self.split_cfg.test,
        )
        n = len(self.dataset)
        train_size = int(train * n)
        test_size = int(test * n)
        val_size = n - train_size - test_size
        pad = self.window_cfg.window - 1 if self.split_cfg.uniform_eval else 0
        idx = range(n)
        self.train_subset = TimeseriesSubset(self.dataset, idx[:train_size])
        self.val_subset = TimeseriesSubset(
            self.dataset, idx[train_size - pad: train_size + val_size]
        )
        self.test_subset = TimeseriesSubset(
            self.dataset, idx[-(test_size + pad):] if pad else idx[-test_size:]
        )

    def _fit_scaler(self) -> None:
        data = self.train_subset.data if self.scale_in_train else self.dataset.data
        self.scaler.fit(data)

    def _build_datasets(self) -> None:
        wc = self.window_cfg
        kw = dict(window=wc.window, stride=wc.stride, time_enc=wc.time_enc, freq=wc.freq)
        self.train_dataset = ImputationWindowedDataset(self.train_subset, self.scaler, **kw)
        self.val_dataset = ImputationWindowedDataset(self.val_subset, self.scaler, **kw)
        self.test_dataset = ImputationWindowedDataset(self.test_subset, self.scaler, **kw)

    def _build_loaders(self) -> None:
        lc = self.loader_cfg
        kw = dict(
            batch_size=lc.batch_size, num_workers=lc.num_workers,
            pin_memory=lc.pin_memory, collate_fn=collate_tsbatch,
            worker_init_fn=seed_worker,
        )
        self.train_loader = DataLoader(self.train_dataset, shuffle=lc.shuffle_train, **kw)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, **kw)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, **kw)

    @property
    def num_features(self) -> int:
        return self.dataset.num_features

    @property
    def window(self) -> int:
        return self.window_cfg.window

    @property
    def mask_ratio(self) -> float:
        return self.window_cfg.mask_ratio
```

- [ ] **Step 4: Add export to `torch_timeseries/dataloader/v2/__init__.py`**

```python
# append after existing imports
from .imputation import ImputationDataModule, ImputationWindowConfig

# add to __all__
__all__ = [
    "TSBatch", "collate_tsbatch", "WindowedDataset",
    "ForecastDataModule", "WindowConfig", "SplitConfig", "LoaderConfig",
    "ImputationDataModule", "ImputationWindowConfig",
    "TimeEncoding",
]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/dataloader/test_v2_task_modules.py -v
```

Expected: 6 PASSED

- [ ] **Step 6: Commit**

```bash
git add torch_timeseries/dataloader/v2/imputation.py torch_timeseries/dataloader/v2/__init__.py tests/dataloader/test_v2_task_modules.py
git commit -m "feat: add ImputationDataModule returning TSBatch"
```

---

### Task 3: AnomalyDataModule

**Files:**
- Create: `torch_timeseries/dataloader/v2/anomaly.py`
- Modify: `torch_timeseries/dataloader/v2/__init__.py`
- Modify: `tests/dataloader/test_v2_task_modules.py`

**Background:** `AnomalyDataset` has `train_data` (all normal data) and `test_data` + `test_labels` (test data with per-timestep binary labels). There is no calendar `dates` in the anomaly split, so `x_time` is None. Train/val split the `train_data` array. Test loader returns `y = labels_for_window` shaped `(B, window)`.

- [ ] **Step 1: Add failing tests to `tests/dataloader/test_v2_task_modules.py`**

```python
# append to tests/dataloader/test_v2_task_modules.py
import numpy as np
from torch_timeseries.core.dataset.anomaly import AnomalyDataset
from torch_timeseries.dataloader.v2.anomaly import AnomalyDataModule, AnomalyWindowConfig


class _ToyAnomaly(AnomalyDataset):
    name = "toy_anomaly"
    num_features = 3
    freq = Freq.hours

    def download(self): pass

    def _load(self):
        rng = np.random.default_rng(42)
        self.train_data = rng.normal(size=(400, 3)).astype(np.float32)
        self.test_data = rng.normal(size=(200, 3)).astype(np.float32)
        self.test_labels = (rng.random(200) > 0.8).astype(np.float32)
        # AnomalyDataset also inherits TimeSeriesDataset fields
        self.data = self.train_data
        self.length = len(self.train_data)
        self.df = pd.DataFrame(self.train_data)
        self.dates = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=len(self.train_data), freq="h")})


def test_anomaly_dm_train_batch_y_is_none():
    dm = AnomalyDataModule(
        dataset=_ToyAnomaly(root="/tmp"),
        scaler=StandardScaler(),
        window=AnomalyWindowConfig(window=20, stride=10),
        loader=LoaderConfig(batch_size=8, num_workers=0),
    )
    batch = next(iter(dm.train_loader))
    assert isinstance(batch, TSBatch)
    assert batch.y is None


def test_anomaly_dm_train_batch_x_shape():
    dm = AnomalyDataModule(
        dataset=_ToyAnomaly(root="/tmp"),
        scaler=StandardScaler(),
        window=AnomalyWindowConfig(window=20, stride=10),
        loader=LoaderConfig(batch_size=8, num_workers=0),
    )
    batch = next(iter(dm.train_loader))
    B = min(8, len(dm.train_dataset))
    assert batch.x.shape == (B, 20, 3)


def test_anomaly_dm_test_batch_y_is_labels():
    dm = AnomalyDataModule(
        dataset=_ToyAnomaly(root="/tmp"),
        scaler=StandardScaler(),
        window=AnomalyWindowConfig(window=20, stride=10),
        loader=LoaderConfig(batch_size=8, num_workers=0),
    )
    batch = next(iter(dm.test_loader))
    assert batch.y is not None
    B = min(8, len(dm.test_dataset))
    assert batch.y.shape == (B, 20)   # per-timestep labels in window
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/dataloader/test_v2_task_modules.py::test_anomaly_dm_train_batch_y_is_none -v
```

Expected: `ModuleNotFoundError: No module named 'torch_timeseries.dataloader.v2.anomaly'`

- [ ] **Step 3: Create `torch_timeseries/dataloader/v2/anomaly.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from torch_timeseries.core.dataset.anomaly import AnomalyDataset
from torch_timeseries.scaler import Scaler

from .._seed import seed_worker
from .batch import TSBatch, collate_tsbatch
from .forecast import LoaderConfig


@dataclass
class AnomalyWindowConfig:
    window: int = 96
    stride: int = 1
    train_ratio: float = 0.8


class AnomalyWindowedDataset(Dataset):
    """Sliding-window dataset from a flat data array (no TimeseriesSubset)."""

    def __init__(
        self,
        data: np.ndarray,
        scaler: Scaler,
        window: int = 96,
        stride: int = 1,
        labels: Optional[np.ndarray] = None,
        scaler_fit: bool = False,
    ) -> None:
        self.window = window
        self.labels = labels
        if scaler_fit:
            scaler.fit(data)
        self.scaled = scaler.transform(data).astype(np.float32)
        self._starts = list(range(0, len(data) - window + 1, stride))

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> TSBatch:
        s = self._starts[idx]
        e = s + self.window
        x = torch.from_numpy(self.scaled[s:e].copy())
        y = None
        if self.labels is not None:
            y = torch.from_numpy(self.labels[s:e].copy())
        return TSBatch(x=x, y=y)


class AnomalyDataModule:
    """Train/val from `dataset.train_data`; test from `dataset.test_data` with labels."""

    def __init__(
        self,
        dataset: AnomalyDataset,
        scaler: Scaler,
        window: AnomalyWindowConfig = None,
        loader: LoaderConfig = None,
    ) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.window_cfg = window or AnomalyWindowConfig()
        self.loader_cfg = loader or LoaderConfig()

        self._build_datasets()
        self._build_loaders()

    def _build_datasets(self) -> None:
        wc = self.window_cfg
        train_end = int(len(self.dataset.train_data) * wc.train_ratio)

        self.train_dataset = AnomalyWindowedDataset(
            self.dataset.train_data[:train_end], self.scaler,
            window=wc.window, stride=wc.stride, scaler_fit=True,
        )
        self.val_dataset = AnomalyWindowedDataset(
            self.dataset.train_data[train_end:], self.scaler,
            window=wc.window, stride=wc.stride,
        )
        self.test_dataset = AnomalyWindowedDataset(
            self.dataset.test_data, self.scaler,
            window=wc.window, stride=wc.stride,
            labels=self.dataset.test_labels,
        )

    def _build_loaders(self) -> None:
        lc = self.loader_cfg
        kw = dict(
            batch_size=lc.batch_size, num_workers=lc.num_workers,
            pin_memory=lc.pin_memory, collate_fn=collate_tsbatch,
            worker_init_fn=seed_worker,
        )
        self.train_loader = DataLoader(self.train_dataset, shuffle=lc.shuffle_train, **kw)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, **kw)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, **kw)

    @property
    def num_features(self) -> int:
        return self.dataset.num_features
```

- [ ] **Step 4: Add export to `torch_timeseries/dataloader/v2/__init__.py`**

Add after the imputation import:

```python
from .anomaly import AnomalyDataModule, AnomalyWindowConfig
```

Add `"AnomalyDataModule"` and `"AnomalyWindowConfig"` to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/dataloader/test_v2_task_modules.py -v -k "anomaly"
```

Expected: 3 PASSED

- [ ] **Step 6: Commit**

```bash
git add torch_timeseries/dataloader/v2/anomaly.py torch_timeseries/dataloader/v2/__init__.py tests/dataloader/test_v2_task_modules.py
git commit -m "feat: add AnomalyDataModule returning TSBatch"
```

---

### Task 4: UEADataModule

**Files:**
- Create: `torch_timeseries/dataloader/v2/uea.py`
- Modify: `torch_timeseries/dataloader/v2/__init__.py`
- Modify: `tests/dataloader/test_v2_task_modules.py`

**Background:** UEA datasets provide pre-split train/test DataFrames (`dataset.train_df`, `dataset.test_df`) and matching label Series (`dataset.train_labels`, `dataset.test_labels`). Each row is one time series sample (multi-index: sample_id × timestep). The DataModule creates a val split from the training indices. Labels are integer class IDs. Returns `TSBatch(x=(T,F), y=scalar_label)`.

- [ ] **Step 1: Add failing tests to `tests/dataloader/test_v2_task_modules.py`**

```python
# append to tests/dataloader/test_v2_task_modules.py
from torch_timeseries.dataloader.v2.uea import UEADataModule, UEAWindowConfig


class _ToyUEA:
    """Minimal stand-in for torch_timeseries.dataset.UEA."""
    name = "toy_uea"
    num_features = 5
    num_classes = 3
    max_seq_len = 24

    def __init__(self, root=None):
        rng = np.random.default_rng(7)
        n_train, n_test, T, F = 60, 20, 24, 5
        # multi-index DataFrame: (sample_id, timestep)
        idx_train = pd.MultiIndex.from_product(
            [range(n_train), range(T)], names=["sample_id", "timestep"]
        )
        idx_test = pd.MultiIndex.from_product(
            [range(n_train, n_train + n_test), range(T)], names=["sample_id", "timestep"]
        )
        self.train_df = pd.DataFrame(
            rng.normal(size=(n_train * T, F)), index=idx_train,
            columns=[f"c{i}" for i in range(F)]
        )
        self.test_df = pd.DataFrame(
            rng.normal(size=(n_test * T, F)), index=idx_test,
            columns=[f"c{i}" for i in range(F)]
        )
        train_label_idx = pd.Index(range(n_train), name="sample_id")
        test_label_idx = pd.Index(range(n_train, n_train + n_test), name="sample_id")
        self.train_labels = pd.Series(
            rng.integers(0, 3, size=n_train), index=train_label_idx
        )
        self.test_labels = pd.Series(
            rng.integers(0, 3, size=n_test), index=test_label_idx
        )


def test_uea_dm_train_batch_shape():
    dm = UEADataModule(
        dataset=_ToyUEA(),
        scaler=StandardScaler(),
        window=UEAWindowConfig(val_ratio=0.2),
        loader=LoaderConfig(batch_size=8, num_workers=0),
    )
    batch = next(iter(dm.train_loader))
    assert isinstance(batch, TSBatch)
    B = min(8, len(dm.train_dataset))
    assert batch.x.shape == (B, 24, 5)   # (B, T, F)
    assert batch.y.shape == (B,)          # integer class labels


def test_uea_dm_test_batch_shape():
    dm = UEADataModule(
        dataset=_ToyUEA(),
        scaler=StandardScaler(),
        window=UEAWindowConfig(val_ratio=0.2),
        loader=LoaderConfig(batch_size=8, num_workers=0),
    )
    batch = next(iter(dm.test_loader))
    B = min(8, len(dm.test_dataset))
    assert batch.x.shape == (B, 24, 5)
    assert batch.y.shape == (B,)


def test_uea_dm_has_val_loader():
    dm = UEADataModule(
        dataset=_ToyUEA(),
        scaler=StandardScaler(),
        window=UEAWindowConfig(val_ratio=0.2),
        loader=LoaderConfig(batch_size=8, num_workers=0),
    )
    assert hasattr(dm, "val_loader")
    assert len(dm.val_dataset) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/dataloader/test_v2_task_modules.py::test_uea_dm_train_batch_shape -v
```

Expected: `ModuleNotFoundError: No module named 'torch_timeseries.dataloader.v2.uea'`

- [ ] **Step 3: Create `torch_timeseries/dataloader/v2/uea.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .._seed import seed_worker
from .batch import TSBatch, collate_tsbatch
from .forecast import LoaderConfig


@dataclass
class UEAWindowConfig:
    val_ratio: float = 0.2
    normalize: bool = True


class UEAWindowedDataset(Dataset):
    """One sample per unique sample_id in the multi-index DataFrame."""

    def __init__(
        self,
        feature_df: pd.DataFrame,
        labels: pd.Series,
        scaler,
        sample_ids,
        scaler_fit: bool = False,
    ) -> None:
        self.sample_ids = list(sample_ids)
        self.feature_df = feature_df
        self.labels = labels
        if scaler_fit:
            scaler.fit(feature_df.loc[self.sample_ids].values)
        self.scaled_df = pd.DataFrame(
            scaler.transform(feature_df), index=feature_df.index,
            columns=feature_df.columns,
        )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> TSBatch:
        sid = self.sample_ids[idx]
        x = torch.from_numpy(self.scaled_df.loc[sid].values.astype(np.float32))
        y = torch.tensor(int(self.labels.loc[sid]), dtype=torch.long)
        return TSBatch(x=x, y=y)


def _collate_uea(samples):
    """collate_tsbatch works for y=scalar by stacking into (B,)."""
    from .batch import collate_tsbatch
    return collate_tsbatch(samples)


class UEADataModule:
    """Wraps a UEA dataset's pre-split train/test DataFrames."""

    def __init__(
        self,
        dataset,
        scaler,
        window: UEAWindowConfig = None,
        loader: LoaderConfig = None,
    ) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.window_cfg = window or UEAWindowConfig()
        self.loader_cfg = loader or LoaderConfig()

        self._build_datasets()
        self._build_loaders()

    def _build_datasets(self) -> None:
        train_ids = list(self.dataset.train_df.index.get_level_values("sample_id").unique())
        val_split = int(len(train_ids) * (1 - self.window_cfg.val_ratio))
        train_ids_tr = train_ids[:val_split]
        train_ids_val = train_ids[val_split:]
        test_ids = list(self.dataset.test_df.index.get_level_values("sample_id").unique())

        all_df = pd.concat([self.dataset.train_df, self.dataset.test_df])
        all_labels = pd.concat([self.dataset.train_labels, self.dataset.test_labels])

        self.train_dataset = UEAWindowedDataset(
            all_df, all_labels, self.scaler, train_ids_tr, scaler_fit=True
        )
        self.val_dataset = UEAWindowedDataset(
            all_df, all_labels, self.scaler, train_ids_val
        )
        self.test_dataset = UEAWindowedDataset(
            all_df, all_labels, self.scaler, test_ids
        )

    def _build_loaders(self) -> None:
        lc = self.loader_cfg
        kw = dict(
            batch_size=lc.batch_size, num_workers=lc.num_workers,
            pin_memory=lc.pin_memory, collate_fn=collate_tsbatch,
            worker_init_fn=seed_worker,
        )
        self.train_loader = DataLoader(self.train_dataset, shuffle=lc.shuffle_train, **kw)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, **kw)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, **kw)

    @property
    def num_features(self) -> int:
        return self.dataset.num_features

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes
```

- [ ] **Step 4: Add export to `torch_timeseries/dataloader/v2/__init__.py`**

Add after the anomaly import:

```python
from .uea import UEADataModule, UEAWindowConfig
```

Add `"UEADataModule"` and `"UEAWindowConfig"` to `__all__`.

- [ ] **Step 5: Run all task module tests**

```bash
pytest tests/dataloader/test_v2_task_modules.py -v
```

Expected: all PASSED (12+ tests)

- [ ] **Step 6: Commit**

```bash
git add torch_timeseries/dataloader/v2/uea.py torch_timeseries/dataloader/v2/__init__.py tests/dataloader/test_v2_task_modules.py
git commit -m "feat: add UEADataModule returning TSBatch"
```

---

### Task 5: Final export wiring + full test run

**Files:**
- Modify: `torch_timeseries/__init__.py`

- [ ] **Step 1: Expose results layer from top-level `__init__.py`**

Open `torch_timeseries/__init__.py`. Current content:

```python
import torch_timeseries.dataset
import torch_timeseries.dataloader
import torch_timeseries.model
from .cli.exp import exp
```

Add:

```python
import torch_timeseries.dataset
import torch_timeseries.dataloader
import torch_timeseries.model
from .cli.exp import exp
from .results import RunResult, LocalBackend, WandbBackend
```

- [ ] **Step 2: Run the full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all previously passing tests still pass, new tests added in this plan all pass.

- [ ] **Step 3: Commit**

```bash
git add torch_timeseries/__init__.py
git commit -m "feat: expose results layer from top-level import"
```
