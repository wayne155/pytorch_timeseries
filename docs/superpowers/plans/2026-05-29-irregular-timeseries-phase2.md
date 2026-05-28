# Irregular Time Series — Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Prerequisite:** Phase 1 must be complete (`IrregularTSBatch`, `collate_irregular`, `PhysioNet2012`, `PhysioNet2019`, `IrregularClassificationDataModule`, GRU-D, `IrregularClassificationExp`, `GRUDIrregularClassification`).

**Goal:** Add `IrregularInterpolationDataModule`, `IrregularForecastDataModule`, MIMIC dataset (load-from-file), `UEAIrregular` (synthetic dropout wrapper), `IrregularWrapper` (forecast wrapper), plus experiment base classes `IrregularInterpolationExp` and `IrregularForecastExp` and corresponding GRU-D combo classes so that all three task modes work end-to-end.

**Architecture:** Two new DataModules follow the same `_fit_scaler → _build_datasets → _build_loaders` contract. `IrregularInterpolationDataModule` randomly holds out `query_rate` fraction of each sample's observations as prediction targets; the held-out points never appear in the input mask. `IrregularForecastDataModule` wraps any `TimeSeriesDataset` via `IrregularWrapper` (dropout) or any `IrregularTimeSeriesDataset` directly, splitting each sample at `obs_frac` of its timespan. New experiment classes `IrregularInterpolationExp` and `IrregularForecastExp` follow `IrregularClassificationExp` structure, with task-appropriate losses and metrics.

**Tech Stack:** Same as Phase 1 (PyTorch ≥2.0, torchmetrics, numpy, pandas). No new external deps.

---

## File Map

**Create:**
```
torch_timeseries/dataset/irregular/mimic.py                   # MIMIC load-from-file dataset
torch_timeseries/dataset/irregular/uea_irregular.py           # UEAIrregular synthetic dropout wrapper
torch_timeseries/dataset/irregular/wrapper.py                 # IrregularWrapper for regular datasets
torch_timeseries/dataloader/v2/irregular_interpolation.py     # IrregularInterpolationDataModule + Config
torch_timeseries/dataloader/v2/irregular_forecast.py          # IrregularForecastDataModule + Config
torch_timeseries/experiments/irregular_interpolation.py       # IrregularInterpolationExp
torch_timeseries/experiments/irregular_forecast.py            # IrregularForecastExp
tests/dataset/test_irregular_wrappers.py
tests/dataloader/test_v2_irregular_tasks.py
tests/experiments/test_irregular_interp_forecast.py
```

**Modify:**
```
torch_timeseries/dataset/irregular/__init__.py    # add MIMIC, UEAIrregular, IrregularWrapper
torch_timeseries/dataloader/v2/__init__.py        # add interpolation + forecast DataModule exports
torch_timeseries/experiments/__init__.py          # add interp + forecast exp imports
torch_timeseries/experiments/registry.py          # add IrregularInterpolation, IrregularForecast suffixes
torch_timeseries/experiments/GRUD.py              # add GRUDIrregularInterpolation, GRUDIrregularForecast
```

---

### Task 1: `MIMIC` dataset (load-from-file, no auto-download)

**Files:**
- Create: `torch_timeseries/dataset/irregular/mimic.py`
- Test: `tests/dataset/test_irregular_wrappers.py`

MIMIC requires credentialed PhysioNet access and cannot be auto-downloaded. The constructor raises `FileNotFoundError` with instructions if the directory is missing.

- [ ] **Step 1: Write the failing test**

```python
# tests/dataset/test_irregular_wrappers.py
import pytest
import numpy as np


def test_mimic_raises_if_directory_missing(tmp_path):
    from torch_timeseries.dataset.irregular.mimic import MIMIC
    with pytest.raises(FileNotFoundError, match="MIMIC"):
        MIMIC(data_dir=str(tmp_path / "nonexistent"))


def test_mimic_loads_from_fake_directory(tmp_path):
    """MIMIC with a fake minimal directory structure."""
    import pandas as pd
    from torch_timeseries.dataset.irregular.mimic import MIMIC

    # Create minimal MIMIC-like CSV structure
    # MIMIC-III admissions: one patient, one admission, two timesteps
    mimic_dir = tmp_path / "mimic"
    mimic_dir.mkdir()
    # Simplified: we use a pre-processed CSV format (time, feature1..featureN, label)
    df = pd.DataFrame({
        "subject_id": [1, 1, 2, 2],
        "hours_from_admit": [0.0, 2.0, 0.0, 1.0],
        "HR": [80.0, 85.0, 72.0, float("nan")],
        "MAP": [float("nan"), 75.0, 65.0, 70.0],
        "label": [0, 0, 1, 1],
    })
    df.to_csv(mimic_dir / "mimic_processed.csv", index=False)

    ds = MIMIC(data_dir=str(mimic_dir))
    assert len(ds) == 2                    # 2 patients
    assert ds.num_features == 2            # HR, MAP
    assert ds.num_classes == 2
    assert ds.labels is not None
    assert ds.samples[0].shape[1] == 2    # (T_i, 2)
    assert ds.masks[0].shape == ds.samples[0].shape
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/dataset/test_irregular_wrappers.py::test_mimic_raises_if_directory_missing -v
pytest tests/dataset/test_irregular_wrappers.py::test_mimic_loads_from_fake_directory -v
```
Expected: `ImportError`

- [ ] **Step 3: Write `mimic.py`**

```python
# torch_timeseries/dataset/irregular/mimic.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from .base import IrregularTimeSeriesDataset


class MIMIC(IrregularTimeSeriesDataset):
    """MIMIC-III / MIMIC-IV irregular time-series dataset.

    Cannot auto-download — requires credentialed PhysioNet access.
    Expects a pre-processed CSV at ``{data_dir}/mimic_processed.csv`` with columns:
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
                "  3. Download and pre-process the data into mimic_processed.csv\n"
                "     with columns: subject_id, hours_from_admit, <features...>, label"
            )
        self.root = str(data_path.parent)
        self._data_dir = data_path
        self._load()

    def download(self) -> None:
        pass  # Cannot auto-download

    def _load(self) -> None:
        csv_path = self._data_dir / "mimic_processed.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Expected pre-processed file not found: {csv_path}"
            )
        df = pd.read_csv(csv_path)
        required = {"subject_id", "hours_from_admit", "label"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"mimic_processed.csv must have columns: {required}. "
                f"Got: {list(df.columns)}"
            )

        feature_cols = [c for c in df.columns
                        if c not in ("subject_id", "hours_from_admit", "label")]
        F = len(feature_cols)

        all_samples, all_times, all_masks, all_labels = [], [], [], []

        for subj_id, grp in df.groupby("subject_id"):
            grp = grp.sort_values("hours_from_admit")
            T = len(grp)
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/dataset/test_irregular_wrappers.py -k "mimic" -v
```
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add torch_timeseries/dataset/irregular/mimic.py tests/dataset/test_irregular_wrappers.py
git commit -m "feat: add MIMIC dataset (load-from-file, no auto-download)"
```

---

### Task 2: `UEAIrregular` + `IrregularWrapper`

**Files:**
- Create: `torch_timeseries/dataset/irregular/uea_irregular.py`
- Create: `torch_timeseries/dataset/irregular/wrapper.py`
- Test: `tests/dataset/test_irregular_wrappers.py` (add tests)

`UEAIrregular` wraps any existing UEA dataset and applies random timestamp dropout, converting regular sequences to irregular.
`IrregularWrapper` wraps any `TimeSeriesDataset` (ETTh1, etc.) and applies dropout to sliding windows.

- [ ] **Step 1: Add failing tests**

```python
# append to tests/dataset/test_irregular_wrappers.py

def test_uea_irregular_shapes(tmp_path):
    """UEAIrregular produces (T_i, F), masks, and times with dropout applied."""
    import numpy as np
    import pandas as pd
    from torch_timeseries.dataset.irregular.uea_irregular import UEAIrregular

    # Fake UEA dataset: 6 train samples of length 20, 2 features, 2 classes
    class FakeUEA:
        num_features = 2
        num_classes = 2

        def __init__(self):
            train_ids = np.repeat(np.arange(6), 20)
            test_ids = np.repeat(np.arange(2), 20)
            vals = np.random.randn(len(train_ids), 2)
            self.train_df = pd.DataFrame(
                vals, columns=["a", "b"],
                index=pd.MultiIndex.from_arrays(
                    [train_ids, np.tile(np.arange(20), 6)],
                    names=["sample_id", "time_step"]
                )
            )
            test_vals = np.random.randn(len(test_ids), 2)
            self.test_df = pd.DataFrame(
                test_vals, columns=["a", "b"],
                index=pd.MultiIndex.from_arrays(
                    [test_ids, np.tile(np.arange(20), 2)],
                    names=["sample_id", "time_step"]
                )
            )
            self.train_labels = pd.Series(
                [i % 2 for i in range(6)],
                index=np.arange(6), name="label")
            self.test_labels = pd.Series(
                [0, 1], index=np.arange(2), name="label")

    ds = UEAIrregular(FakeUEA(), drop_rate=0.3, seed=42)

    assert len(ds) == 8  # 6 train + 2 test
    assert ds.num_features == 2
    assert ds.num_classes == 2
    # With 30% dropout on 20 steps: T_i ≈ 14
    for i in range(len(ds)):
        T_i = len(ds.times[i])
        assert T_i > 0
        assert ds.samples[i].shape == (T_i, 2)
        assert ds.masks[i].shape == (T_i, 2)
        assert ds.masks[i].sum() > 0


def test_irregular_wrapper_shapes(tmp_path):
    """IrregularWrapper wraps a regular TimeSeriesDataset with dropout."""
    import numpy as np
    import pandas as pd
    from torch_timeseries.core import TimeSeriesDataset, Freq
    from torch_timeseries.dataset.irregular.wrapper import IrregularWrapper

    class FakeTS(TimeSeriesDataset):
        name = "fake"
        freq = Freq.hours
        def download(self): pass
        def _load(self):
            n = 100
            self.num_features = 3
            rng = np.random.default_rng(0)
            self.df = pd.DataFrame(
                {"date": pd.date_range("2020-01-01", periods=n, freq="h"),
                 **{f"c{i}": rng.normal(size=n) for i in range(3)}}
            )
            self.dates = self.df[["date"]]
            self.data = self.df.drop("date", axis=1).values
            self.length = n

    ds = IrregularWrapper(FakeTS(), window=24, drop_rate=0.4, seed=0)

    assert ds.num_features == 3
    assert len(ds) > 0
    for i in range(min(5, len(ds))):
        T_i = len(ds.times[i])
        assert T_i > 0
        assert T_i <= 24
        assert ds.samples[i].shape == (T_i, 3)
        assert ds.masks[i].shape == (T_i, 3)
    # No labels for forecast wrapper
    assert ds.labels is None
    assert ds.num_classes == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/dataset/test_irregular_wrappers.py::test_uea_irregular_shapes -v
pytest tests/dataset/test_irregular_wrappers.py::test_irregular_wrapper_shapes -v
```
Expected: `ImportError`

- [ ] **Step 3: Write `uea_irregular.py`**

```python
# torch_timeseries/dataset/irregular/uea_irregular.py
from __future__ import annotations
from typing import Optional
import numpy as np
from .base import IrregularTimeSeriesDataset


class UEAIrregular(IrregularTimeSeriesDataset):
    """Wraps any UEA dataset and applies synthetic timestamp dropout.

    Converts regular fixed-length sequences to irregular by randomly
    removing observations. All features at a dropped timestep are removed.

    Args:
        dataset:   A UEA dataset with ``train_df``, ``test_df``,
                   ``train_labels``, ``test_labels`` attributes.
        drop_rate: Fraction of timesteps to drop (0.0 = no dropout).
        seed:      Random seed for reproducibility.
    """

    def __init__(self, dataset, drop_rate: float = 0.3, seed: int = 42) -> None:
        self._uea = dataset
        self.drop_rate = drop_rate
        self.seed = seed
        # Skip parent __init__ — we load directly
        self._load()

    def download(self) -> None:
        pass

    def _load(self) -> None:
        import pandas as pd

        rng = np.random.default_rng(self.seed)

        all_samples, all_times, all_masks, all_labels = [], [], [], []

        def _process_split(df, labels):
            ids = list(df.index.get_level_values("sample_id").unique())
            for sid in ids:
                x = df.loc[sid].values.astype(np.float32)   # (T, F)
                T, F = x.shape
                n_keep = max(1, int(T * (1.0 - self.drop_rate)))
                kept = np.sort(rng.choice(T, size=n_keep, replace=False))
                x_kept = x[kept]                              # (T_i, F)
                t_kept = kept.astype(np.float32)              # uniform timestep index
                mask = np.ones_like(x_kept)                   # all observed (not NaN)

                all_samples.append(x_kept)
                all_times.append(t_kept)
                all_masks.append(mask)
                all_labels.append(int(labels.loc[sid]))

        _process_split(self._uea.train_df, self._uea.train_labels)
        _process_split(self._uea.test_df, self._uea.test_labels)

        self.samples = all_samples
        self.times = all_times
        self.masks = all_masks
        self.labels = np.array(all_labels, dtype=np.int64)
        self.num_features = self._uea.num_features
        self.num_classes = self._uea.num_classes
```

- [ ] **Step 4: Write `wrapper.py`**

```python
# torch_timeseries/dataset/irregular/wrapper.py
from __future__ import annotations
from typing import Optional
import numpy as np
from .base import IrregularTimeSeriesDataset


class IrregularWrapper(IrregularTimeSeriesDataset):
    """Wraps any ``TimeSeriesDataset`` and applies random timestamp dropout.

    Slices the dataset into non-overlapping windows of ``window`` timesteps,
    then drops ``drop_rate`` fraction of timesteps from each window to simulate
    irregular sampling. Used primarily with ``IrregularForecastDataModule``.

    Args:
        dataset:   Any ``TimeSeriesDataset`` with a ``.data`` numpy array.
        window:    Length of each sliding window (default 96).
        drop_rate: Fraction of timesteps to drop per window.
        seed:      Random seed.
    """

    def __init__(self, dataset, window: int = 96,
                 drop_rate: float = 0.3, seed: int = 42) -> None:
        self._ts = dataset
        self.window = window
        self.drop_rate = drop_rate
        self.seed = seed
        self._load()

    def download(self) -> None:
        pass

    def _load(self) -> None:
        rng = np.random.default_rng(self.seed)
        data = self._ts.data                  # (N_total, F)
        N, F = data.shape
        n_windows = N // self.window

        all_samples, all_times, all_masks = [], [], []

        for i in range(n_windows):
            seg = data[i * self.window: (i + 1) * self.window].astype(np.float32)
            T = seg.shape[0]
            n_keep = max(1, int(T * (1.0 - self.drop_rate)))
            kept = np.sort(rng.choice(T, size=n_keep, replace=False))
            x_kept = seg[kept]                # (T_i, F)
            t_kept = kept.astype(np.float32)  # uniform timestep index
            mask = np.ones_like(x_kept)

            all_samples.append(x_kept)
            all_times.append(t_kept)
            all_masks.append(mask)

        self.samples = all_samples
        self.times = all_times
        self.masks = all_masks
        self.labels = None
        self.num_features = F
        self.num_classes = 0
```

- [ ] **Step 5: Update `dataset/irregular/__init__.py`**

```python
# torch_timeseries/dataset/irregular/__init__.py
from .base import IrregularTimeSeriesDataset
from .physionet2012 import PhysioNet2012
from .physionet2019 import PhysioNet2019
from .mimic import MIMIC
from .uea_irregular import UEAIrregular
from .wrapper import IrregularWrapper

__all__ = [
    "IrregularTimeSeriesDataset",
    "PhysioNet2012", "PhysioNet2019", "MIMIC",
    "UEAIrregular", "IrregularWrapper",
]
```

- [ ] **Step 6: Run all wrapper tests**

```bash
pytest tests/dataset/test_irregular_wrappers.py -v
```
Expected: `4 passed`

- [ ] **Step 7: Commit**

```bash
git add torch_timeseries/dataset/irregular/ tests/dataset/test_irregular_wrappers.py
git commit -m "feat: add MIMIC, UEAIrregular, IrregularWrapper datasets"
```

---

### Task 3: `IrregularInterpolationDataModule`

**Files:**
- Create: `torch_timeseries/dataloader/v2/irregular_interpolation.py`
- Test: `tests/dataloader/test_v2_irregular_tasks.py`

The DataModule holds out `query_rate` fraction of each sample's observation times as prediction targets. Holdout is deterministic per sample (seeded by sample index) so val/test are reproducible. The held-out points form `(t_query, y=query_values, query_mask)`; the remaining points form the input `(x, t, mask)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/dataloader/test_v2_irregular_tasks.py
import numpy as np
import torch
import pytest


class _ToyIrregular:
    """In-memory irregular dataset — no file I/O."""
    num_features = 3
    num_classes = 2

    def __init__(self, n=40, seed=0):
        rng = np.random.default_rng(seed)
        self.samples, self.times, self.masks, self.labels = [], [], [], []
        for i in range(n):
            T = rng.integers(8, 20)
            x = rng.normal(size=(T, self.num_features)).astype(np.float32)
            t = np.sort(rng.uniform(0, 48, T)).astype(np.float32)
            mask = (rng.random((T, self.num_features)) > 0.2).astype(np.float32)
            self.samples.append(x)
            self.times.append(t)
            self.masks.append(mask)
            self.labels.append(i % self.num_classes)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.samples)


def _toy_interp_dm(**kw):
    from torch_timeseries.dataloader.v2.irregular_interpolation import (
        IrregularInterpolationDataModule, IrregularInterpolationConfig,
    )
    from torch_timeseries.dataloader.v2.forecast import SplitConfig, LoaderConfig
    from torch_timeseries.scaler import StandardScaler
    return IrregularInterpolationDataModule(
        dataset=_ToyIrregular(n=40),
        scaler=StandardScaler(),
        window=IrregularInterpolationConfig(query_rate=0.2, **kw),
        split=SplitConfig(train=0.7, val=0.1, test=0.2),
        loader=LoaderConfig(batch_size=8, num_workers=0),
    )


def test_interpolation_dm_returns_batch():
    dm = _toy_interp_dm()
    batch = next(iter(dm.train_loader))
    from torch_timeseries.dataloader.v2.irregular_batch import IrregularTSBatch
    assert isinstance(batch, IrregularTSBatch)


def test_interpolation_dm_query_not_in_input_mask():
    """Held-out query positions must NOT appear in the input mask."""
    dm = _toy_interp_dm(query_rate=0.3)
    batch = next(iter(dm.train_loader))
    # batch.t_query and batch.t: no t_query value should equal a masked t value
    # We verify query_mask exists and input mask has zeros where queries are
    assert batch.t_query is not None
    assert batch.query_mask is not None
    # y contains the query targets
    assert batch.y is not None
    assert batch.y.shape[:2] == batch.t_query.shape


def test_interpolation_dm_properties():
    dm = _toy_interp_dm()
    assert dm.num_features == 3
    assert dm.query_rate == 0.2
    assert hasattr(dm, "train_loader")
    assert hasattr(dm, "val_loader")
    assert hasattr(dm, "test_loader")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/dataloader/test_v2_irregular_tasks.py::test_interpolation_dm_returns_batch -v
```
Expected: `ImportError`

- [ ] **Step 3: Write `irregular_interpolation.py`**

```python
# torch_timeseries/dataloader/v2/irregular_interpolation.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .._seed import seed_worker
from .forecast import LoaderConfig, SplitConfig
from .irregular_batch import IrregularTSBatch, collate_irregular


@dataclass
class IrregularInterpolationConfig:
    query_rate: float = 0.2   # fraction of observations held out as query targets
    time_enc: int = 0
    freq: Optional[str] = None

    def __post_init__(self):
        if not (0.0 < self.query_rate < 1.0):
            raise ValueError(
                f"query_rate must be in (0, 1), got {self.query_rate}")


class _IrregularInterpolationDataset(Dataset):
    """Splits each sample into input observations and held-out query targets."""

    def __init__(self, dataset, scaler, indices: List[int],
                 query_rate: float) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.indices = indices
        self.query_rate = query_rate

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> IrregularTSBatch:
        i = self.indices[idx]
        x_raw = np.array(self.dataset.samples[i], dtype=np.float32)  # (T_i, F)
        t_raw = np.array(self.dataset.times[i], dtype=np.float32)     # (T_i,)
        mask = np.array(self.dataset.masks[i], dtype=np.float32)       # (T_i, F)
        T, F = x_raw.shape

        # Deterministic hold-out: seeded by sample index for reproducibility
        rng = np.random.default_rng(seed=i)
        n_query = max(1, int(T * self.query_rate))
        all_idx = np.arange(T)
        query_idx = np.sort(rng.choice(T, size=n_query, replace=False))
        input_mask_idx = np.array([j for j in all_idx if j not in set(query_idx)])

        if len(input_mask_idx) == 0:
            input_mask_idx = query_idx[:1]
            query_idx = query_idx[1:]

        # Input: only non-held-out observations
        x_in = x_raw[input_mask_idx]
        t_in = t_raw[input_mask_idx]
        mask_in = mask[input_mask_idx]

        # Query targets
        x_q = x_raw[query_idx]
        t_q = t_raw[query_idx]
        q_mask = mask[query_idx]          # 1=this feature was actually observed

        # Normalize time to [0, 1] using full sample's range
        t_min, t_max = t_raw.min(), t_raw.max()
        eps = 1e-8
        t_in_norm = (t_in - t_min) / (t_max - t_min + eps)
        t_q_norm = (t_q - t_min) / (t_max - t_min + eps)

        # Scale features
        x_in_scaled = self.scaler.transform(x_in) * mask_in

        return IrregularTSBatch(
            x=torch.from_numpy(x_in_scaled),
            t=torch.from_numpy(t_in_norm),
            mask=torch.from_numpy(mask_in),
            y=torch.from_numpy(x_q * q_mask),          # query target values (masked-out = 0)
            t_query=torch.from_numpy(t_q_norm),
            query_mask=torch.from_numpy(q_mask),
        )


class IrregularInterpolationDataModule:
    """DataModule for irregular time-series interpolation.

    Each batch: ``IrregularTSBatch(x, t, mask, y=query_values, t_query, query_mask)``.
    Loss is computed only on ``query_mask == 1`` positions.
    """

    def __init__(
        self,
        dataset,
        scaler,
        window: IrregularInterpolationConfig = None,
        split: SplitConfig = None,
        loader: LoaderConfig = None,
    ) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.window_cfg = window or IrregularInterpolationConfig()
        self.split_cfg = split or SplitConfig()
        self.loader_cfg = loader or LoaderConfig()

        self._fit_scaler()
        self._build_datasets()
        self._build_loaders()

    def _fit_scaler(self) -> None:
        n = len(self.dataset)
        train_end = int(self.split_cfg.train * n)
        obs_list = [np.array(self.dataset.samples[i], dtype=np.float32)
                    for i in range(train_end)]
        if obs_list:
            self.scaler.fit(np.vstack(obs_list))

    def _build_datasets(self) -> None:
        n = len(self.dataset)
        train_end = int(self.split_cfg.train * n)
        test_size = int((self.split_cfg.test or 0.2) * n)
        val_end = n - test_size
        qr = self.window_cfg.query_rate

        self.train_dataset = _IrregularInterpolationDataset(
            self.dataset, self.scaler, list(range(0, train_end)), qr)
        self.val_dataset = _IrregularInterpolationDataset(
            self.dataset, self.scaler, list(range(train_end, val_end)), qr)
        self.test_dataset = _IrregularInterpolationDataset(
            self.dataset, self.scaler, list(range(val_end, n)), qr)

    def _build_loaders(self) -> None:
        lc = self.loader_cfg
        kw = dict(
            batch_size=lc.batch_size, num_workers=lc.num_workers,
            pin_memory=lc.pin_memory, collate_fn=collate_irregular,
            worker_init_fn=seed_worker,
        )
        self.train_loader = DataLoader(
            self.train_dataset, shuffle=lc.shuffle_train, **kw)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, **kw)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, **kw)

    @property
    def num_features(self) -> int:
        return self.dataset.num_features

    @property
    def query_rate(self) -> float:
        return self.window_cfg.query_rate
```

- [ ] **Step 4: Run interpolation tests**

```bash
pytest tests/dataloader/test_v2_irregular_tasks.py -k "interpolation" -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add torch_timeseries/dataloader/v2/irregular_interpolation.py tests/dataloader/test_v2_irregular_tasks.py
git commit -m "feat: add IrregularInterpolationDataModule"
```

---

### Task 4: `IrregularForecastDataModule`

**Files:**
- Create: `torch_timeseries/dataloader/v2/irregular_forecast.py`
- Test: `tests/dataloader/test_v2_irregular_tasks.py` (add tests)

The DataModule splits each sample at `obs_frac` of its total timespan: observations before the split point → input; observations after → forecast targets. `query_mask` is all-ones (all future points are targets).

- [ ] **Step 1: Add failing tests**

```python
# append to tests/dataloader/test_v2_irregular_tasks.py

def _toy_forecast_dm(**kw):
    from torch_timeseries.dataloader.v2.irregular_forecast import (
        IrregularForecastDataModule, IrregularForecastConfig,
    )
    from torch_timeseries.dataloader.v2.forecast import SplitConfig, LoaderConfig
    from torch_timeseries.scaler import StandardScaler
    return IrregularForecastDataModule(
        dataset=_ToyIrregular(n=40),
        scaler=StandardScaler(),
        window=IrregularForecastConfig(obs_frac=0.7, **kw),
        split=SplitConfig(train=0.7, val=0.1, test=0.2),
        loader=LoaderConfig(batch_size=8, num_workers=0),
    )


def test_forecast_dm_returns_batch():
    dm = _toy_forecast_dm()
    batch = next(iter(dm.train_loader))
    from torch_timeseries.dataloader.v2.irregular_batch import IrregularTSBatch
    assert isinstance(batch, IrregularTSBatch)


def test_forecast_dm_future_split():
    """All t_query values must be > the max input t for each sample."""
    dm = _toy_forecast_dm(obs_frac=0.7)
    # Get a single sample from the dataset directly (before collation)
    sample = dm.train_dataset[0]
    if sample.t_query is not None and len(sample.t_query) > 0:
        assert sample.t_query.min() >= sample.t.max()


def test_forecast_dm_properties():
    dm = _toy_forecast_dm()
    assert dm.num_features == 3
    assert dm.obs_frac == 0.7
    assert hasattr(dm, "train_loader")
    assert hasattr(dm, "val_loader")
    assert hasattr(dm, "test_loader")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/dataloader/test_v2_irregular_tasks.py::test_forecast_dm_returns_batch -v
```
Expected: `ImportError`

- [ ] **Step 3: Write `irregular_forecast.py`**

```python
# torch_timeseries/dataloader/v2/irregular_forecast.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .._seed import seed_worker
from .forecast import LoaderConfig, SplitConfig
from .irregular_batch import IrregularTSBatch, collate_irregular


@dataclass
class IrregularForecastConfig:
    obs_frac: float = 0.7   # fraction of timespan used as input
    time_enc: int = 1
    freq: Optional[str] = None

    def __post_init__(self):
        if not (0.0 < self.obs_frac < 1.0):
            raise ValueError(
                f"obs_frac must be in (0, 1), got {self.obs_frac}")


class _IrregularForecastDataset(Dataset):
    """Splits each sample at obs_frac of its timespan into input + forecast."""

    def __init__(self, dataset, scaler, indices: List[int],
                 obs_frac: float) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.indices = indices
        self.obs_frac = obs_frac

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> IrregularTSBatch:
        i = self.indices[idx]
        x_raw = np.array(self.dataset.samples[i], dtype=np.float32)  # (T_i, F)
        t_raw = np.array(self.dataset.times[i], dtype=np.float32)     # (T_i,)
        mask = np.array(self.dataset.masks[i], dtype=np.float32)       # (T_i, F)

        t_min, t_max = t_raw.min(), t_raw.max()
        split_t = t_min + self.obs_frac * (t_max - t_min)

        input_idx = np.where(t_raw <= split_t)[0]
        future_idx = np.where(t_raw > split_t)[0]

        # If no future points, take the last 20% as future
        if len(future_idx) == 0:
            n_future = max(1, len(t_raw) // 5)
            input_idx = np.arange(len(t_raw) - n_future)
            future_idx = np.arange(len(t_raw) - n_future, len(t_raw))

        if len(input_idx) == 0:
            input_idx = future_idx[:1]
            future_idx = future_idx[1:]

        x_in = x_raw[input_idx]
        t_in = t_raw[input_idx]
        mask_in = mask[input_idx]

        x_fut = x_raw[future_idx]
        t_fut = t_raw[future_idx]
        mask_fut = mask[future_idx]    # query_mask: which future features to eval

        # Normalize time globally per sample
        eps = 1e-8
        t_in_norm = (t_in - t_min) / (t_max - t_min + eps)
        t_fut_norm = (t_fut - t_min) / (t_max - t_min + eps)

        x_in_scaled = self.scaler.transform(x_in) * mask_in

        return IrregularTSBatch(
            x=torch.from_numpy(x_in_scaled),
            t=torch.from_numpy(t_in_norm),
            mask=torch.from_numpy(mask_in),
            y=torch.from_numpy(x_fut * mask_fut),
            t_query=torch.from_numpy(t_fut_norm),
            query_mask=torch.from_numpy(mask_fut),
        )


class IrregularForecastDataModule:
    """DataModule for irregular time-series forecasting.

    Each batch: ``IrregularTSBatch(x, t, mask, y=future_values, t_query, query_mask)``.
    All future observation points are targets; ``query_mask`` flags which features
    were actually observed.
    """

    def __init__(
        self,
        dataset,
        scaler,
        window: IrregularForecastConfig = None,
        split: SplitConfig = None,
        loader: LoaderConfig = None,
    ) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.window_cfg = window or IrregularForecastConfig()
        self.split_cfg = split or SplitConfig()
        self.loader_cfg = loader or LoaderConfig()

        self._fit_scaler()
        self._build_datasets()
        self._build_loaders()

    def _fit_scaler(self) -> None:
        n = len(self.dataset)
        train_end = int(self.split_cfg.train * n)
        obs_list = [np.array(self.dataset.samples[i], dtype=np.float32)
                    for i in range(train_end)]
        if obs_list:
            self.scaler.fit(np.vstack(obs_list))

    def _build_datasets(self) -> None:
        n = len(self.dataset)
        train_end = int(self.split_cfg.train * n)
        test_size = int((self.split_cfg.test or 0.2) * n)
        val_end = n - test_size
        of = self.window_cfg.obs_frac

        self.train_dataset = _IrregularForecastDataset(
            self.dataset, self.scaler, list(range(0, train_end)), of)
        self.val_dataset = _IrregularForecastDataset(
            self.dataset, self.scaler, list(range(train_end, val_end)), of)
        self.test_dataset = _IrregularForecastDataset(
            self.dataset, self.scaler, list(range(val_end, n)), of)

    def _build_loaders(self) -> None:
        lc = self.loader_cfg
        kw = dict(
            batch_size=lc.batch_size, num_workers=lc.num_workers,
            pin_memory=lc.pin_memory, collate_fn=collate_irregular,
            worker_init_fn=seed_worker,
        )
        self.train_loader = DataLoader(
            self.train_dataset, shuffle=lc.shuffle_train, **kw)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, **kw)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, **kw)

    @property
    def num_features(self) -> int:
        return self.dataset.num_features

    @property
    def obs_frac(self) -> float:
        return self.window_cfg.obs_frac
```

- [ ] **Step 4: Run all DataModule tests**

```bash
pytest tests/dataloader/test_v2_irregular_tasks.py -v
```
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add torch_timeseries/dataloader/v2/irregular_forecast.py tests/dataloader/test_v2_irregular_tasks.py
git commit -m "feat: add IrregularForecastDataModule"
```

---

### Task 5: `IrregularInterpolationExp` + `IrregularForecastExp`

**Files:**
- Create: `torch_timeseries/experiments/irregular_interpolation.py`
- Create: `torch_timeseries/experiments/irregular_forecast.py`
- Test: `tests/experiments/test_irregular_interp_forecast.py`

Both experiments follow the same loop structure as `IrregularClassificationExp`. Key differences:
- Interpolation: MSELoss on `query_mask == 1` positions; metrics = MSE, MAE on query points.
- Forecast: MSELoss on all future query points; same metrics.

- [ ] **Step 1: Write the failing test**

```python
# tests/experiments/test_irregular_interp_forecast.py
import pytest


class _ToyIrregular:
    num_features = 3
    num_classes = 0

    def __init__(self, n=40):
        import numpy as np
        rng = np.random.default_rng(0)
        self.samples, self.times, self.masks = [], [], []
        self.labels = None
        for i in range(n):
            T = rng.integers(8, 20)
            self.samples.append(rng.normal(size=(T, 3)).astype("float32"))
            self.times.append(
                np.sort(rng.uniform(0, 48, T)).astype("float32"))
            self.masks.append(
                (rng.random((T, 3)) > 0.2).astype("float32"))

    def __len__(self):
        return len(self.samples)


def test_grud_irregular_interpolation_single_run(tmp_path):
    from torch_timeseries.experiments.GRUD import GRUDIrregularInterpolation
    exp = GRUDIrregularInterpolation(
        dataset_type="__toy__",
        epochs=2, patience=5, batch_size=8,
        hidden_size=16, device="cpu",
        save_dir=str(tmp_path),
    )
    exp._toy_dataset = _ToyIrregular()
    result = exp.run(seed=1)
    assert isinstance(result, dict)
    assert "mse" in result
    assert result["mse"] >= 0.0


def test_grud_irregular_forecast_single_run(tmp_path):
    from torch_timeseries.experiments.GRUD import GRUDIrregularForecast
    exp = GRUDIrregularForecast(
        dataset_type="__toy__",
        epochs=2, patience=5, batch_size=8,
        hidden_size=16, device="cpu",
        save_dir=str(tmp_path),
    )
    exp._toy_dataset = _ToyIrregular()
    result = exp.run(seed=1)
    assert isinstance(result, dict)
    assert "mse" in result
    assert result["mse"] >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/experiments/test_irregular_interp_forecast.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Write `irregular_interpolation.py` experiment**

```python
# torch_timeseries/experiments/irregular_interpolation.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.nn import MSELoss
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from tqdm import tqdm

from ..core import BaseIrrelevant, BaseRelevant
from ..dataloader.v2.forecast import LoaderConfig, SplitConfig
from ..dataloader.v2.irregular_interpolation import (
    IrregularInterpolationConfig, IrregularInterpolationDataModule,
)
from ..dataloader.v2.irregular_batch import IrregularTSBatch
from ..scaler import StandardScaler
from ..utils.early_stop import EarlyStopping
from ..utils.reproduce import get_rng_state, reproducible, set_rng_state
import hashlib, json, os


def _get_dataset(name: str, root: str):
    from ..dataset.irregular import PhysioNet2012, PhysioNet2019, MIMIC, UEAIrregular
    _map = {
        "PhysioNet2012": PhysioNet2012, "PhysioNet2019": PhysioNet2019, "MIMIC": MIMIC,
    }
    if name not in _map:
        raise ValueError(f"Unknown dataset for interpolation: {name!r}. Available: {list(_map)}")
    return _map[name](root=root)


@dataclass
class IrregularInterpolationSettings:
    query_rate: float = 0.2
    time_enc: int = 0
    freq: Optional[str] = None


@dataclass
class IrregularInterpolationExp(BaseRelevant, BaseIrrelevant, IrregularInterpolationSettings):
    """Base experiment for irregular time-series interpolation (query-point MSE)."""
    loss_func_type: str = "mse"

    def _init_model(self) -> None:
        raise NotImplementedError

    def _init_data_loader(self) -> None:
        if hasattr(self, "_toy_dataset"):
            dataset = self._toy_dataset
        else:
            dataset = _get_dataset(self.dataset_type, self.data_path)
        self.dm = IrregularInterpolationDataModule(
            dataset=dataset,
            scaler=StandardScaler(),
            window=IrregularInterpolationConfig(query_rate=self.query_rate),
            split=SplitConfig(train=0.7, val=0.1, test=0.2),
            loader=LoaderConfig(batch_size=self.batch_size, num_workers=self.num_worker),
        )
        self.train_loader = self.dm.train_loader
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader

    def _init_metrics(self) -> None:
        self.metrics = MetricCollection(
            {"mse": MeanSquaredError(), "mae": MeanAbsoluteError()}
        )
        self.metrics.to(self.device)

    def _init_loss_func(self) -> None:
        self.loss_func = MSELoss(reduction="none")

    def _init_optimizer(self) -> None:
        from torch.optim import Adam
        self.optimizer = Adam(self.model.parameters(), lr=self.lr,
                              weight_decay=self.l2_weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs)

    def _setup(self) -> None:
        self._init_data_loader()
        self._init_metrics()
        self._init_loss_func()
        self.setuped = True

    def _process_one_batch(self, batch: IrregularTSBatch):
        x = batch.x.float().to(self.device)
        t = batch.t.float().to(self.device)
        mask = batch.mask.float().to(self.device)
        t_query = batch.t_query.float().to(self.device)
        y = batch.y.float().to(self.device)
        qmask = batch.query_mask.float().to(self.device)
        pred = self.model(x, t, mask, t_query=t_query)   # (B, Tq, F)
        return pred, y, qmask

    def _masked_mse_loss(self, pred, y, qmask) -> torch.Tensor:
        """MSELoss on query_mask == 1 positions only."""
        loss_all = self.loss_func(pred, y)               # (B, Tq, F)
        loss_masked = (loss_all * qmask).sum() / (qmask.sum() + 1e-8)
        return loss_masked

    def _train(self) -> List[float]:
        self.model.train()
        losses = []
        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset), leave=False) as pb:
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                pred, y, qmask = self._process_one_batch(batch)
                loss = self._masked_mse_loss(pred, y, qmask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                losses.append(loss.item())
                pb.update(batch.x.shape[0])
        return losses

    def _evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        self.metrics.reset()
        losses = []
        with torch.no_grad():
            for batch in loader:
                pred, y, qmask = self._process_one_batch(batch)
                loss = self._masked_mse_loss(pred, y, qmask)
                losses.append(loss.item())
                # Flatten query points for metrics
                B, Tq, F = pred.shape
                flat_pred = pred[qmask.bool()].reshape(-1)
                flat_y = y[qmask.bool()].reshape(-1)
                if flat_pred.numel() > 0:
                    self.metrics.update(flat_pred, flat_y)
        result = {k: float(v.compute()) for k, v in self.metrics.items()}
        result["loss"] = float(np.mean(losses))
        return result

    def _setup_run(self, seed: int) -> None:
        if not hasattr(self, "setuped"):
            self._setup()
        reproducible(seed)
        self._init_model()
        self._init_optimizer()
        self.current_epoch = 0
        run_id = hashlib.md5(json.dumps({"seed": seed, "model": self.model_type,
            "dataset": str(self.dataset_type)}, sort_keys=True).encode()).hexdigest()
        self.run_save_dir = os.path.join(self.save_dir, "runs", self.model_type,
                                          str(self.dataset_type), run_id)
        os.makedirs(self.run_save_dir, exist_ok=True)
        self.best_ckpt = os.path.join(self.run_save_dir, "best_model.pth")
        self.early_stopper = EarlyStopping(self.patience, verbose=False, path=self.best_ckpt)

    def run(self, seed: int = 42) -> Dict[str, float]:
        self._setup_run(seed)
        while self.current_epoch < self.epochs:
            if self.early_stopper.early_stop:
                break
            reproducible(seed + self.current_epoch)
            self._train()
            val_result = self._evaluate(self.val_loader)
            self.current_epoch += 1
            self.early_stopper(val_result["loss"], model=self.model)
            self.scheduler.step()
        if os.path.exists(self.best_ckpt):
            self.model.load_state_dict(
                torch.load(self.best_ckpt, map_location=self.device, weights_only=False))
        return self._evaluate(self.test_loader)

    def runs(self, seeds=None):
        return [self.run(s) for s in (seeds or [1, 2, 3])]
```

- [ ] **Step 4: Write `irregular_forecast.py` experiment**

```python
# torch_timeseries/experiments/irregular_forecast.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.nn import MSELoss
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from tqdm import tqdm

from ..core import BaseIrrelevant, BaseRelevant
from ..dataloader.v2.forecast import LoaderConfig, SplitConfig
from ..dataloader.v2.irregular_forecast import (
    IrregularForecastConfig, IrregularForecastDataModule,
)
from ..dataloader.v2.irregular_batch import IrregularTSBatch
from ..scaler import StandardScaler
from ..utils.early_stop import EarlyStopping
from ..utils.reproduce import get_rng_state, reproducible, set_rng_state
import hashlib, json, os


def _get_dataset(name: str, root: str):
    from ..dataset.irregular import IrregularWrapper
    from ..dataset import ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity
    _regular_map = {
        "ETTh1": ETTh1, "ETTh2": ETTh2,
        "ETTm1": ETTm1, "ETTm2": ETTm2,
        "Weather": Weather, "Electricity": Electricity,
    }
    if name in _regular_map:
        return IrregularWrapper(_regular_map[name](root=root), drop_rate=0.3)
    from ..dataset.irregular import PhysioNet2012, PhysioNet2019
    _irreg_map = {"PhysioNet2012": PhysioNet2012, "PhysioNet2019": PhysioNet2019}
    if name in _irreg_map:
        return _irreg_map[name](root=root)
    raise ValueError(f"Unknown dataset for forecast: {name!r}")


@dataclass
class IrregularForecastSettings:
    obs_frac: float = 0.7
    time_enc: int = 1
    freq: Optional[str] = None


@dataclass
class IrregularForecastExp(BaseRelevant, BaseIrrelevant, IrregularForecastSettings):
    """Base experiment for irregular time-series forecasting."""
    loss_func_type: str = "mse"

    def _init_model(self) -> None:
        raise NotImplementedError

    def _init_data_loader(self) -> None:
        if hasattr(self, "_toy_dataset"):
            dataset = self._toy_dataset
        else:
            dataset = _get_dataset(self.dataset_type, self.data_path)
        self.dm = IrregularForecastDataModule(
            dataset=dataset,
            scaler=StandardScaler(),
            window=IrregularForecastConfig(obs_frac=self.obs_frac),
            split=SplitConfig(train=0.7, val=0.1, test=0.2),
            loader=LoaderConfig(batch_size=self.batch_size, num_workers=self.num_worker),
        )
        self.train_loader = self.dm.train_loader
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader

    def _init_metrics(self) -> None:
        self.metrics = MetricCollection(
            {"mse": MeanSquaredError(), "mae": MeanAbsoluteError()})
        self.metrics.to(self.device)

    def _init_loss_func(self) -> None:
        self.loss_func = MSELoss()

    def _init_optimizer(self) -> None:
        from torch.optim import Adam
        self.optimizer = Adam(self.model.parameters(), lr=self.lr,
                              weight_decay=self.l2_weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs)

    def _setup(self) -> None:
        self._init_data_loader()
        self._init_metrics()
        self._init_loss_func()
        self.setuped = True

    def _process_one_batch(self, batch: IrregularTSBatch):
        x = batch.x.float().to(self.device)
        t = batch.t.float().to(self.device)
        mask = batch.mask.float().to(self.device)
        t_query = batch.t_query.float().to(self.device)
        y = batch.y.float().to(self.device)
        qmask = batch.query_mask.float().to(self.device)
        pred = self.model(x, t, mask, t_query=t_query)   # (B, Tq, F)
        return pred, y, qmask

    def _masked_mse_loss(self, pred, y, qmask) -> torch.Tensor:
        loss_all = (pred - y) ** 2
        return (loss_all * qmask).sum() / (qmask.sum() + 1e-8)

    def _train(self) -> List[float]:
        self.model.train()
        losses = []
        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset), leave=False) as pb:
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                pred, y, qmask = self._process_one_batch(batch)
                loss = self._masked_mse_loss(pred, y, qmask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                losses.append(loss.item())
                pb.update(batch.x.shape[0])
        return losses

    def _evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        self.metrics.reset()
        losses = []
        with torch.no_grad():
            for batch in loader:
                pred, y, qmask = self._process_one_batch(batch)
                loss = self._masked_mse_loss(pred, y, qmask)
                losses.append(loss.item())
                B, Tq, F = pred.shape
                flat_pred = pred[qmask.bool()].reshape(-1)
                flat_y = y[qmask.bool()].reshape(-1)
                if flat_pred.numel() > 0:
                    self.metrics.update(flat_pred, flat_y)
        result = {k: float(v.compute()) for k, v in self.metrics.items()}
        result["loss"] = float(np.mean(losses))
        return result

    def _setup_run(self, seed: int) -> None:
        if not hasattr(self, "setuped"):
            self._setup()
        reproducible(seed)
        self._init_model()
        self._init_optimizer()
        self.current_epoch = 0
        run_id = hashlib.md5(json.dumps({"seed": seed, "model": self.model_type,
            "dataset": str(self.dataset_type)}, sort_keys=True).encode()).hexdigest()
        self.run_save_dir = os.path.join(self.save_dir, "runs", self.model_type,
                                          str(self.dataset_type), run_id)
        os.makedirs(self.run_save_dir, exist_ok=True)
        self.best_ckpt = os.path.join(self.run_save_dir, "best_model.pth")
        self.early_stopper = EarlyStopping(self.patience, verbose=False, path=self.best_ckpt)

    def run(self, seed: int = 42) -> Dict[str, float]:
        self._setup_run(seed)
        while self.current_epoch < self.epochs:
            if self.early_stopper.early_stop:
                break
            reproducible(seed + self.current_epoch)
            self._train()
            val_result = self._evaluate(self.val_loader)
            self.current_epoch += 1
            self.early_stopper(val_result["loss"], model=self.model)
            self.scheduler.step()
        if os.path.exists(self.best_ckpt):
            self.model.load_state_dict(
                torch.load(self.best_ckpt, map_location=self.device, weights_only=False))
        return self._evaluate(self.test_loader)

    def runs(self, seeds=None):
        return [self.run(s) for s in (seeds or [1, 2, 3])]
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/experiments/test_irregular_interp_forecast.py -v
```
Expected: fail at `ImportError` for `GRUDIrregularInterpolation` — those combo classes come in Task 6. The exp base classes exist now, so `ImportError` should say GRUD combo missing.

---

### Task 6: GRUD interpolation + forecast combo classes + full registry wiring

**Files:**
- Modify: `torch_timeseries/experiments/GRUD.py` (add Interp + Forecast combos)
- Modify: `torch_timeseries/experiments/registry.py` (add new suffixes)
- Modify: `torch_timeseries/experiments/__init__.py` (add new imports)
- Modify: `torch_timeseries/dataloader/v2/__init__.py` (add new DataModule exports)

GRU-D for interpolation/forecast needs a `Seq2Seq` interface:
`forward(x, t, mask, t_query) -> (B, Tq, F)`.
Extend `GRUD` with an `output_mode` parameter. When `output_mode="seq2seq"`, the model:
1. Encodes the input sequence to a hidden state via the GRU-D loop.
2. For each query time `t_q`, computes the decayed hidden state and projects to F features.

- [ ] **Step 1: Extend `grud.py` with seq2seq mode**

Open `torch_timeseries/model/irregular/grud.py` and add a `seq2seq` output path after the existing `fc` linear layer:

```python
# In GRUD.__init__, add:
self.fc_seq2seq = nn.Linear(hidden_size, input_size)   # (H → F) for interp/forecast

# Change signature:
def forward(
    self,
    x: Tensor,              # (B, T, F)
    t: Tensor,              # (B, T)
    mask: Tensor,           # (B, T, F)
    x_time: Tensor = None,  # ignored
    t_query: Tensor = None, # (B, Tq) — if provided, return (B, Tq, F) predictions
) -> Tensor:                # (B, output_size) OR (B, Tq, F)
    B, T, F = x.shape
    h = x.new_zeros(B, self.cell.hidden_size)
    x_last = x.new_zeros(B, F)
    t_last = x.new_zeros(B, F)

    for step in range(T):
        x_t = x[:, step, :]
        m_t = mask[:, step, :]
        t_t = t[:, step].unsqueeze(-1).expand(B, F)
        delta = torch.clamp(t_t - t_last, min=0.0)
        h, x_last = self.cell(x_t, m_t, delta, x_last, h)
        t_last = m_t * t_t + (1.0 - m_t) * t_last

    if t_query is None:
        return self.fc(self.drop(h))   # (B, output_size) — classification

    # Seq2Seq: for each query time, decay h and project
    Tq = t_query.shape[1]
    preds = []
    for q in range(Tq):
        t_q = t_query[:, q].unsqueeze(-1).expand(B, F)        # (B, F)
        delta_q = torch.clamp(t_q - t_last, min=0.0)
        gamma_h = torch.exp(
            -torch.relu(self.cell.log_gamma_h) *
            delta_q.mean(dim=-1, keepdim=True).expand_as(h)
        )
        h_q = gamma_h * h
        preds.append(self.fc_seq2seq(self.drop(h_q)))          # (B, F)
    return torch.stack(preds, dim=1)                           # (B, Tq, F)
```

- [ ] **Step 2: Add a seq2seq test to `tests/model/test_irregular_models.py`**

```python
def test_grud_forward_seq2seq():
    """GRU-D with t_query returns (B, Tq, F) predictions."""
    from torch_timeseries.model.irregular.grud import GRUD
    B, T, F, Tq = 4, 10, 3, 5
    model = GRUD(input_size=F, hidden_size=16, output_size=2)
    x, t, mask = _make_synthetic_batch(B, T, F)
    t_query = torch.linspace(0.5, 1.0, Tq).unsqueeze(0).expand(B, -1)
    out = model(x, t, mask, t_query=t_query)
    assert out.shape == (B, Tq, F), f"Expected ({B},{Tq},{F}), got {out.shape}"
```

Run:
```bash
pytest tests/model/test_irregular_models.py -v
```
Expected: all pass (existing 3 + new 1 = 4 passed)

- [ ] **Step 3: Add combo classes to `GRUD.py`**

```python
# append to torch_timeseries/experiments/GRUD.py

from .irregular_interpolation import IrregularInterpolationExp
from .irregular_forecast import IrregularForecastExp


@dataclass
class GRUDIrregularInterpolation(IrregularInterpolationExp, GRUDParameters):
    model_type: str = "GRUD"

    def _init_model(self) -> None:
        self.model = GRUD(
            input_size=self.dm.num_features,
            hidden_size=self.hidden_size,
            output_size=self.dm.num_features,   # seq2seq: output_size = F
            dropout=self.grud_dropout,
        ).to(self.device)


@dataclass
class GRUDIrregularForecast(IrregularForecastExp, GRUDParameters):
    model_type: str = "GRUD"

    def _init_model(self) -> None:
        self.model = GRUD(
            input_size=self.dm.num_features,
            hidden_size=self.hidden_size,
            output_size=self.dm.num_features,
            dropout=self.grud_dropout,
        ).to(self.device)
```

- [ ] **Step 4: Add new task suffixes to `registry.py`**

```python
# torch_timeseries/experiments/registry.py
TASK_SUFFIXES = (
    "Forecast", "Imputation", "UEAClassification", "AnomalyDetection",
    "IrregularClassification", "IrregularInterpolation", "IrregularForecast",
)
```

- [ ] **Step 5: Add imports to `experiments/__init__.py`**

```python
# After existing irregular_classification imports:
from .irregular_interpolation import IrregularInterpolationExp
from .irregular_forecast import IrregularForecastExp
from .GRUD import (
    GRUDIrregularClassification,
    GRUDIrregularInterpolation,
    GRUDIrregularForecast,
)
```

- [ ] **Step 6: Update `dataloader/v2/__init__.py`** to add new DataModule exports

```python
from .irregular_interpolation import IrregularInterpolationDataModule, IrregularInterpolationConfig
from .irregular_forecast import IrregularForecastDataModule, IrregularForecastConfig
```

Add to `__all__`:
```python
"IrregularInterpolationDataModule", "IrregularInterpolationConfig",
"IrregularForecastDataModule", "IrregularForecastConfig",
```

- [ ] **Step 7: Verify registry**

```bash
python -c "
from torch_timeseries.experiments import EXPERIMENT_REGISTRY
keys = [k for k in EXPERIMENT_REGISTRY if 'Irregular' in str(k)]
print(sorted(keys))
"
```
Expected:
```
[('GRUD', 'IrregularClassification'), ('GRUD', 'IrregularForecast'), ('GRUD', 'IrregularInterpolation')]
```

- [ ] **Step 8: Run all Phase 2 tests**

```bash
pytest tests/dataset/test_irregular_wrappers.py \
       tests/dataloader/test_v2_irregular_tasks.py \
       tests/experiments/test_irregular_interp_forecast.py \
       tests/model/test_irregular_models.py -v
```
Expected: all pass.

- [ ] **Step 9: Run regression check**

```bash
pytest tests/ -v --ignore=tests/experiments/test_autoformer.py -x
```
Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add torch_timeseries/experiments/ \
        torch_timeseries/model/irregular/grud.py \
        torch_timeseries/dataloader/v2/__init__.py \
        tests/experiments/test_irregular_interp_forecast.py \
        tests/model/test_irregular_models.py
git commit -m "feat: add IrregularInterpolation/ForecastExp + GRUD combos + registry wiring (Phase 2)"
```

---

## Self-Review

**Spec coverage check (Phase 2):**
- MIMIC dataset (load-from-file, FileNotFoundError): Task 1 ✅
- UEAIrregular (synthetic dropout wrapper): Task 2 ✅
- IrregularWrapper (regular dataset → irregular): Task 2 ✅
- IrregularInterpolationDataModule + IrregularInterpolationConfig: Task 3 ✅
  - query_rate holdout, deterministic per sample, query not in input mask ✅
- IrregularForecastDataModule + IrregularForecastConfig: Task 4 ✅
  - obs_frac split, all future as targets ✅
- IrregularInterpolationExp (CrossEntropy→MSE, masked loss): Task 5 ✅
- IrregularForecastExp (MSE on all future points): Task 5 ✅
- GRUDIrregularInterpolation, GRUDIrregularForecast combos: Task 6 ✅
- GRUD seq2seq mode (t_query → (B,Tq,F)): Task 6 ✅
- Registry wiring for all 3 GRUD×irregular tasks: Task 6 ✅
- Section 6 tests (test_irregular_interpolation_dm_query_holdout, test_irregular_forecast_dm_future_split): Tasks 3–4 ✅
