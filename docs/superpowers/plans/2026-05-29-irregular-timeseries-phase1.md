# Irregular Time Series — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `IrregularTSBatch`, PhysioNet2012/2019 datasets, `IrregularClassificationDataModule`, GRU-D model, and `IrregularClassificationExp` + GRUD combo class so that `Experiment(model="GRUD", task="IrregularClassification", dataset="PhysioNet2012").run(seeds=[1])` works end-to-end.

**Architecture:** New classes follow the existing dataloader-v2 / experiment patterns. `IrregularTSBatch` replaces `TSBatch` for irregular data; `collate_irregular` pads variable-length sequences. `IrregularClassificationDataModule` follows the `_fit_scaler → _build_datasets → _build_loaders` contract. `IrregularClassificationExp` follows `UEAClassificationExp` loop structure but uses the v2 DataModule and `IrregularTSBatch`. GRU-D handles temporal decay for missing values.

**Tech Stack:** Python ≥3.8, PyTorch ≥2.0, torchmetrics ≥1.0, numpy, pandas, urllib (stdlib), pytest ≥7.0

---

## File Map

**Create:**
```
torch_timeseries/dataloader/v2/irregular_batch.py     # IrregularTSBatch, collate_irregular
torch_timeseries/dataset/irregular/__init__.py        # package + re-exports
torch_timeseries/dataset/irregular/base.py            # IrregularTimeSeriesDataset base class
torch_timeseries/dataset/irregular/physionet2012.py   # PhysioNet2012
torch_timeseries/dataset/irregular/physionet2019.py   # PhysioNet2019
torch_timeseries/dataloader/v2/irregular_classification.py  # IrregularClassificationDataModule + Config
torch_timeseries/model/irregular/__init__.py          # model package
torch_timeseries/model/irregular/grud.py              # GRU-D model
torch_timeseries/experiments/irregular_classification.py    # IrregularClassificationExp
torch_timeseries/experiments/GRUD.py                  # GRUDIrregularClassification combo
tests/dataloader/test_v2_irregular.py
tests/dataset/test_irregular_datasets.py
tests/model/test_irregular_models.py
tests/experiments/test_irregular_experiments.py
```

**Modify:**
```
torch_timeseries/dataloader/v2/__init__.py            # add IrregularTSBatch exports
torch_timeseries/experiments/__init__.py              # import GRUD combo
pyproject.toml                                        # add [irregular] optional deps
```

---

### Task 1: `IrregularTSBatch` dataclass + `collate_irregular`

**Files:**
- Create: `torch_timeseries/dataloader/v2/irregular_batch.py`
- Test: `tests/dataloader/test_v2_irregular.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dataloader/test_v2_irregular.py
import torch
import pytest

def test_irregular_tsbatch_collation():
    """Variable-length batch pads correctly; mask shape correct; t=1.0 at padded positions."""
    from torch_timeseries.dataloader.v2.irregular_batch import IrregularTSBatch, collate_irregular
    F = 3
    samples = []
    for T in [4, 6, 5]:
        x = torch.randn(T, F)
        t = torch.linspace(0.0, 1.0, T)
        mask = (torch.rand(T, F) > 0.3).float()
        y = torch.tensor(0, dtype=torch.long)
        samples.append(IrregularTSBatch(x=x, t=t, mask=mask, y=y))

    batch = collate_irregular(samples)

    assert batch.x.shape == (3, 6, F)
    assert batch.t.shape == (3, 6)
    assert batch.mask.shape == (3, 6, F)
    assert batch.y.shape == (3,)
    # sample 0 (T=4): positions 4 and 5 must be padded → mask=0
    assert batch.mask[0, 4:, :].sum() == 0
    # sample 1 (T=6): no padding → has real observations
    assert batch.mask[1, :, :].sum() > 0
    # padded t positions must be 1.0
    assert (batch.t[0, 4:] == 1.0).all()


def test_collate_irregular_with_query_times():
    """Query-time fields (for interp/forecast) are also padded correctly."""
    from torch_timeseries.dataloader.v2.irregular_batch import IrregularTSBatch, collate_irregular
    F = 2
    samples = []
    for T, Tq in [(4, 2), (6, 3), (5, 2)]:
        x = torch.randn(T, F)
        t = torch.linspace(0.0, 1.0, T)
        mask = torch.ones(T, F)
        y = torch.randn(Tq, F)
        t_query = torch.linspace(0.5, 1.0, Tq)
        query_mask = torch.ones(Tq, F)
        samples.append(IrregularTSBatch(x=x, t=t, mask=mask, y=y,
                                         t_query=t_query, query_mask=query_mask))

    batch = collate_irregular(samples)
    assert batch.t_query.shape == (3, 3)
    assert batch.query_mask.shape == (3, 3, F)
    # sample 0 and 2 (Tq=2): position 2 padded → query_mask=0
    assert batch.query_mask[0, 2, :].sum() == 0
    assert batch.query_mask[2, 2, :].sum() == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/dataloader/test_v2_irregular.py -v
```
Expected: `ImportError` or `ModuleNotFoundError` — `irregular_batch` does not exist yet.

- [ ] **Step 3: Write `irregular_batch.py`**

```python
# torch_timeseries/dataloader/v2/irregular_batch.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor


@dataclass
class IrregularTSBatch:
    x: Tensor                               # (B, T, F)  padded observed values
    t: Tensor                               # (B, T)     elapsed time, normalized [0,1]
    mask: Tensor                            # (B, T, F)  1=observed, 0=missing/padded
    x_time: Optional[Tensor] = None        # (B, T, C)  calendar features at obs times
    y: Optional[Tensor] = None             # (B,) class label  OR  (B, Tq, F) query targets
    t_query: Optional[Tensor] = None       # (B, Tq)    query times (interp / forecast)
    query_mask: Optional[Tensor] = None    # (B, Tq, F) which queries to evaluate
    t_query_time: Optional[Tensor] = None  # (B, Tq, C) calendar features at query times


def collate_irregular(samples: List[IrregularTSBatch]) -> IrregularTSBatch:
    """Pad variable-length sequences to the longest sequence in the batch."""
    max_T = max(s.x.shape[0] for s in samples)
    F = samples[0].x.shape[1]

    xs, ts, masks = [], [], []
    for s in samples:
        T_i = s.x.shape[0]
        x_pad = torch.zeros(max_T, F)
        x_pad[:T_i] = s.x
        xs.append(x_pad)

        t_pad = torch.ones(max_T)
        t_pad[:T_i] = s.t
        ts.append(t_pad)

        m_pad = torch.zeros(max_T, F)
        m_pad[:T_i] = s.mask
        masks.append(m_pad)

    x_batch = torch.stack(xs, dim=0)
    t_batch = torch.stack(ts, dim=0)
    mask_batch = torch.stack(masks, dim=0)

    # --- x_time (calendar at observation times) ---
    x_time_batch = None
    if samples[0].x_time is not None:
        C = samples[0].x_time.shape[1]
        x_times = []
        for s in samples:
            T_i = s.x_time.shape[0]
            xt = torch.zeros(max_T, C)
            xt[:T_i] = s.x_time
            x_times.append(xt)
        x_time_batch = torch.stack(x_times, dim=0)

    # --- y (class label scalar OR query-target tensor) ---
    y_batch = None
    if samples[0].y is not None:
        if samples[0].y.dim() == 0:
            # scalar labels
            y_batch = torch.stack([s.y for s in samples], dim=0)
        else:
            # variable-length query targets → pad like x
            max_Tq_y = max(s.y.shape[0] for s in samples)
            Fy = samples[0].y.shape[1]
            ys = []
            for s in samples:
                Tq_i = s.y.shape[0]
                yp = torch.zeros(max_Tq_y, Fy)
                yp[:Tq_i] = s.y
                ys.append(yp)
            y_batch = torch.stack(ys, dim=0)

    # --- query times (interpolation / forecast) ---
    t_query_batch = query_mask_batch = t_query_time_batch = None
    if samples[0].t_query is not None:
        max_Tq = max(s.t_query.shape[0] for s in samples)
        Fq = samples[0].query_mask.shape[1] if samples[0].query_mask is not None else F
        tqs, qms = [], []
        for s in samples:
            Tq_i = s.t_query.shape[0]
            tq = torch.ones(max_Tq)
            tq[:Tq_i] = s.t_query
            tqs.append(tq)
            if s.query_mask is not None:
                qm = torch.zeros(max_Tq, Fq)
                qm[:Tq_i] = s.query_mask
                qms.append(qm)
        t_query_batch = torch.stack(tqs, dim=0)
        if qms:
            query_mask_batch = torch.stack(qms, dim=0)

        if samples[0].t_query_time is not None:
            C2 = samples[0].t_query_time.shape[1]
            tqts = []
            for s in samples:
                Tq_i = s.t_query_time.shape[0]
                tqt = torch.zeros(max_Tq, C2)
                tqt[:Tq_i] = s.t_query_time
                tqts.append(tqt)
            t_query_time_batch = torch.stack(tqts, dim=0)

    return IrregularTSBatch(
        x=x_batch, t=t_batch, mask=mask_batch,
        x_time=x_time_batch, y=y_batch,
        t_query=t_query_batch, query_mask=query_mask_batch,
        t_query_time=t_query_time_batch,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/dataloader/test_v2_irregular.py -v
```
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add torch_timeseries/dataloader/v2/irregular_batch.py tests/dataloader/test_v2_irregular.py
git commit -m "feat: add IrregularTSBatch dataclass and collate_irregular"
```

---

### Task 2: `IrregularTimeSeriesDataset` base class + dataset package

**Files:**
- Create: `torch_timeseries/dataset/irregular/__init__.py`
- Create: `torch_timeseries/dataset/irregular/base.py`

No separate test for the abstract base (tested via concrete datasets in Task 3).

- [ ] **Step 1: Write `base.py`**

```python
# torch_timeseries/dataset/irregular/base.py
from __future__ import annotations
from typing import List, Optional
import numpy as np


class IrregularTimeSeriesDataset:
    """Base class for irregular time-series datasets.

    Subclasses implement ``download()`` and ``_load()``.
    After ``_load()``, the following attributes must be set:

        samples:      List[np.ndarray]  — [(T_i, F)]  variable-length per sample
        times:        List[np.ndarray]  — [(T_i,)]    raw observation times (any unit)
        masks:        List[np.ndarray]  — [(T_i, F)]  1=observed, 0=missing
        labels:       Optional[np.ndarray]  — (N,) integer class labels; None if no labels
        num_features: int
        num_classes:  int               — 0 if no labels
    """

    samples: List[np.ndarray]
    times: List[np.ndarray]
    masks: List[np.ndarray]
    labels: Optional[np.ndarray]
    num_features: int
    num_classes: int

    def __init__(self, root: str, download: bool = True) -> None:
        self.root = root
        if download:
            self.download()
        self._load()

    def download(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} must implement download()")

    def _load(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} must implement _load()")

    def __len__(self) -> int:
        return len(self.samples)
```

- [ ] **Step 2: Write `__init__.py`**

```python
# torch_timeseries/dataset/irregular/__init__.py
from .base import IrregularTimeSeriesDataset
from .physionet2012 import PhysioNet2012
from .physionet2019 import PhysioNet2019

__all__ = ["IrregularTimeSeriesDataset", "PhysioNet2012", "PhysioNet2019"]
```

- [ ] **Step 3: Verify import (no test yet — that comes in Task 3)**

```bash
python -c "from torch_timeseries.dataset.irregular.base import IrregularTimeSeriesDataset; print('ok')"
```
Expected: `ok` (will fail until physionet2012.py / physionet2019.py exist)

- [ ] **Step 4: Commit base + placeholder `__init__.py` (with physionet imports commented out)**

For now, comment out the PhysioNet imports in `__init__.py` until Tasks 3–4:
```python
# torch_timeseries/dataset/irregular/__init__.py
from .base import IrregularTimeSeriesDataset
# populated in Tasks 3-4:
# from .physionet2012 import PhysioNet2012
# from .physionet2019 import PhysioNet2019
__all__ = ["IrregularTimeSeriesDataset"]
```

```bash
git add torch_timeseries/dataset/irregular/
git commit -m "feat: add IrregularTimeSeriesDataset base class and dataset package"
```

---

### Task 3: `PhysioNet2012` dataset

**Files:**
- Create: `torch_timeseries/dataset/irregular/physionet2012.py`
- Test: `tests/dataset/test_irregular_datasets.py`

PhysioNet 2012 challenge format:
- Per-patient files in `{root}/physionet2012/set-a/NNNNNN.txt`:
  - Header lines (e.g. `RecordID,142738`), then blank line, then `Time,Parameter,Value` table.
  - Time is `HH:MM`, Parameter is a variable name, Value is a float.
- Outcome file at `{root}/physionet2012/Outcomes-a.txt` with columns:
  `RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death`
- 41 variables total (see `PhysioNet2012.VARIABLES` list).

The test creates a minimal fake directory and verifies loading without any network access.

- [ ] **Step 1: Write the failing test**

```python
# tests/dataset/test_irregular_datasets.py
import numpy as np
import pytest


def _write_fake_physionet2012(tmp_path, record_ids, outcomes):
    """Create minimal PhysioNet 2012 set-a directory."""
    set_a = tmp_path / "physionet2012" / "set-a"
    set_a.mkdir(parents=True)
    for rec_id in record_ids:
        lines = [
            f"RecordID,{rec_id}",
            "Age,50", "Gender,1", "Height,170", "ICUType,1", "Weight,70",
            "",
            "Time,Parameter,Value",
            "00:07,HR,109",
            "00:07,GCS,15",
            "01:35,HR,122",
            "02:00,Temp,37.2",
        ]
        (set_a / f"{rec_id}.txt").write_text("\n".join(lines))
    outcome_lines = [
        "RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death"
    ]
    for rec_id, label in zip(record_ids, outcomes):
        outcome_lines.append(f"{rec_id},6,1,5,3950.0,{label}")
    (tmp_path / "physionet2012" / "Outcomes-a.txt").write_text("\n".join(outcome_lines))


def test_physionet2012_loads(tmp_path):
    _write_fake_physionet2012(tmp_path, [140501, 140936, 141091], [0, 1, 0])
    from torch_timeseries.dataset.irregular import PhysioNet2012
    ds = PhysioNet2012(root=str(tmp_path), download=False)

    assert len(ds) == 3
    assert ds.num_features == 41
    assert ds.num_classes == 2
    assert ds.labels is not None
    assert len(ds.labels) == 3
    # each sample is (T_i, 41)
    assert ds.samples[0].ndim == 2
    assert ds.samples[0].shape[1] == 41
    # times: (T_i,)
    assert ds.times[0].ndim == 1
    assert len(ds.times[0]) == ds.samples[0].shape[0]
    # masks: (T_i, 41), binary
    assert ds.masks[0].shape == ds.samples[0].shape
    assert set(np.unique(ds.masks[0])).issubset({0, 1})
    # HR is observed → at least one mask entry is 1
    assert ds.masks[0].sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/dataset/test_irregular_datasets.py::test_physionet2012_loads -v
```
Expected: `ImportError` — `PhysioNet2012` does not exist yet.

- [ ] **Step 3: Write `physionet2012.py`**

```python
# torch_timeseries/dataset/irregular/physionet2012.py
from __future__ import annotations
import os
import urllib.request
import tarfile
from pathlib import Path
from typing import Optional
import numpy as np

from .base import IrregularTimeSeriesDataset


class PhysioNet2012(IrregularTimeSeriesDataset):
    """PhysioNet Challenge 2012 — in-hospital mortality (binary classification).

    12,000 ICU patient records from set-a (4,000), set-b (4,000), set-c (4,000).
    41 time-varying variables; up to 48 hours at irregular intervals.
    Label: ``In-hospital_death`` (0/1).

    Auto-downloads from https://physionet.org/files/challenge-2012/1.0.0/
    if data is not already present at ``{root}/physionet2012/``.
    """

    # 37 time-varying variables + 4 static (Age, Gender, Height, ICUType)
    # included as fixed-time observations at t=0
    VARIABLES = [
        "Age", "Gender", "Height", "ICUType",
        "ALP", "ALT", "AST", "Albumin", "BUN",
        "Bicarbonate", "Bilirubin", "Cholesterol", "Creatinine",
        "DiasABP", "FiO2", "GCS", "Glucose", "HCT", "HR",
        "K", "Lactate", "MAP", "MechVent", "Mg", "Na",
        "NIDiasABP", "NIMAP", "NISysABP", "PaCO2", "PaO2",
        "Platelets", "RespRate", "SaO2", "SysABP", "Temp",
        "TroponinI", "TroponinT", "Urine", "WBC", "Weight", "pH",
    ]
    _VAR_IDX = {v: i for i, v in enumerate(VARIABLES)}

    _BASE_URL = "https://physionet.org/files/challenge-2012/1.0.0/"
    _SETS = ["a", "b", "c"]

    def download(self) -> None:
        dest = Path(self.root) / "physionet2012"
        if (dest / "set-a").exists():
            return
        dest.mkdir(parents=True, exist_ok=True)
        for s in self._SETS:
            tar_url = f"{self._BASE_URL}set-{s}.tar.gz"
            tar_path = dest / f"set-{s}.tar.gz"
            print(f"Downloading {tar_url} ...")
            urllib.request.urlretrieve(tar_url, tar_path)
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(dest)
            tar_path.unlink()
            outcome_url = f"{self._BASE_URL}Outcomes-{s}.txt"
            urllib.request.urlretrieve(outcome_url, dest / f"Outcomes-{s}.txt")

    def _load(self) -> None:
        dest = Path(self.root) / "physionet2012"
        all_samples, all_times, all_masks, all_labels = [], [], [], []
        F = len(self.VARIABLES)

        for s in self._SETS:
            set_dir = dest / f"set-{s}"
            outcome_path = dest / f"Outcomes-{s}.txt"
            if not set_dir.exists() or not outcome_path.exists():
                continue

            # Parse outcomes
            outcomes = {}
            with open(outcome_path) as f:
                next(f)  # skip header
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 6:
                        try:
                            outcomes[int(parts[0])] = int(parts[5])
                        except ValueError:
                            pass

            for txt_file in sorted(set_dir.glob("*.txt")):
                try:
                    rec_id, x, t_arr, mask = self._parse_patient(txt_file, F)
                except Exception:
                    continue
                if rec_id not in outcomes:
                    continue
                all_samples.append(x)
                all_times.append(t_arr)
                all_masks.append(mask)
                all_labels.append(outcomes[rec_id])

        self.samples = all_samples
        self.times = all_times
        self.masks = all_masks
        self.labels = np.array(all_labels, dtype=np.int64)
        self.num_features = F
        self.num_classes = 2

    def _parse_patient(self, path: Path, F: int):
        """Parse one PhysioNet 2012 patient file.

        Returns (rec_id, x, t_arr, mask) where:
            x:     (T_unique, F) float32 — observed values (0 where missing)
            t_arr: (T_unique,)  float32 — minutes since admission
            mask:  (T_unique, F) float32 — 1=observed
        """
        header = {}
        obs = {}  # minute → {var_idx: value}

        in_header = True
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    in_header = False
                    continue
                if in_header:
                    k, v = line.split(",", 1)
                    header[k] = v
                else:
                    if line.startswith("Time,"):
                        continue
                    parts = line.split(",")
                    if len(parts) != 3:
                        continue
                    hhmm, param, val_str = parts
                    if param not in self._VAR_IDX:
                        continue
                    try:
                        hh, mm = hhmm.split(":")
                        minutes = int(hh) * 60 + int(mm)
                        val = float(val_str)
                    except ValueError:
                        continue
                    if minutes not in obs:
                        obs[minutes] = {}
                    obs[minutes][self._VAR_IDX[param]] = val

        rec_id = int(header.get("RecordID", -1))

        # Add static header variables at t=0
        for svar in ["Age", "Gender", "Height", "ICUType"]:
            if svar in header:
                try:
                    val = float(header[svar])
                    if val >= 0:  # -1 means missing in PhysioNet 2012
                        if 0 not in obs:
                            obs[0] = {}
                        obs[0][self._VAR_IDX[svar]] = val
                except ValueError:
                    pass

        if not obs:
            # Empty patient — create single zero timestep
            obs[0] = {}

        sorted_times = sorted(obs.keys())
        T = len(sorted_times)
        x = np.zeros((T, F), dtype=np.float32)
        mask = np.zeros((T, F), dtype=np.float32)
        t_arr = np.array(sorted_times, dtype=np.float32)

        for i, minute in enumerate(sorted_times):
            for var_idx, val in obs[minute].items():
                x[i, var_idx] = val
                mask[i, var_idx] = 1.0

        return rec_id, x, t_arr, mask
```

- [ ] **Step 4: Uncomment `PhysioNet2012` import in `dataset/irregular/__init__.py`**

```python
# torch_timeseries/dataset/irregular/__init__.py
from .base import IrregularTimeSeriesDataset
from .physionet2012 import PhysioNet2012
# PhysioNet2019 added in Task 4:
# from .physionet2019 import PhysioNet2019
__all__ = ["IrregularTimeSeriesDataset", "PhysioNet2012"]
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/dataset/test_irregular_datasets.py::test_physionet2012_loads -v
```
Expected: `1 passed`

- [ ] **Step 6: Commit**

```bash
git add torch_timeseries/dataset/irregular/ tests/dataset/test_irregular_datasets.py
git commit -m "feat: add PhysioNet2012 dataset with fake-data test"
```

---

### Task 4: `PhysioNet2019` dataset

**Files:**
- Create: `torch_timeseries/dataset/irregular/physionet2019.py`
- Test: `tests/dataset/test_irregular_datasets.py` (add a new test)

PhysioNet 2019 (sepsis prediction) format:
- Patient files are PSV (pipe-separated) with a header row of variable names.
- Each row: one timestep (hourly), columns include `ICULOS` (ICU length of stay in hours) and 40 clinical variables.
- Label is in the last column `SepsisLabel` (0/1).
- Two sets: `training_setA/` and `training_setB/`.

- [ ] **Step 1: Add the failing test**

```python
# append to tests/dataset/test_irregular_datasets.py

def _write_fake_physionet2019(tmp_path, patient_ids, labels):
    """Create minimal PhysioNet 2019 set-A directory."""
    set_a = tmp_path / "physionet2019" / "training_setA"
    set_a.mkdir(parents=True)
    P19_HEADER = (
        "HR|O2Sat|Temp|SBP|MAP|DBP|Resp|EtCO2|BaseExcess|HCO3|FiO2|pH|"
        "PaCO2|SaO2|AST|BUN|Alkalinephos|Calcium|Chloride|Creatinine|"
        "Bilirubin_direct|Glucose|Lactate|Magnesium|Phosphate|Potassium|"
        "Bilirubin_total|TroponinI|Hct|Hgb|PTT|WBC|Fibrinogen|Platelets|"
        "Age|Gender|Unit1|Unit2|HospAdmTime|ICULOS|SepsisLabel"
    )
    for pid, label in zip(patient_ids, labels):
        lines = [P19_HEADER]
        lines.append(f"80|98|37.0|120|80|60|16|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|40|NaN|NaN|NaN|NaN|NaN|65|0|0|0|-5|1|0")
        lines.append(f"85|97|37.2|118|78|58|18|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|40|NaN|NaN|NaN|NaN|NaN|65|0|0|0|-5|2|{label}")
        (set_a / f"p{pid:06d}.psv").write_text("\n".join(lines))


def test_physionet2019_loads(tmp_path):
    _write_fake_physionet2019(tmp_path, [1, 2, 3], [0, 1, 0])
    from torch_timeseries.dataset.irregular import PhysioNet2019
    ds = PhysioNet2019(root=str(tmp_path), download=False)

    assert len(ds) == 3
    assert ds.num_features == 40
    assert ds.num_classes == 2
    assert ds.labels is not None
    assert len(ds.labels) == 3
    assert ds.samples[0].ndim == 2
    assert ds.samples[0].shape[1] == 40
    assert ds.times[0].ndim == 1
    assert ds.masks[0].shape == ds.samples[0].shape
    assert set(np.unique(ds.masks[0])).issubset({0, 1})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/dataset/test_irregular_datasets.py::test_physionet2019_loads -v
```
Expected: `ImportError`

- [ ] **Step 3: Write `physionet2019.py`**

```python
# torch_timeseries/dataset/irregular/physionet2019.py
from __future__ import annotations
import urllib.request
import zipfile
from pathlib import Path
import numpy as np

from .base import IrregularTimeSeriesDataset


class PhysioNet2019(IrregularTimeSeriesDataset):
    """PhysioNet Challenge 2019 — sepsis onset prediction (binary classification).

    ~40,000 ICU patient records, 40 clinical variables, hourly observations.
    Label: ``SepsisLabel`` (0/1).

    Auto-downloads from PhysioNet Challenge 2019 public page if not present.
    """

    # 40 clinical variables (the 41st column in the PSV is SepsisLabel, excluded here)
    VARIABLES = [
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
        "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
        "Fibrinogen", "Platelets",
        "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
    ]
    _VAR_IDX = {v: i for i, v in enumerate(VARIABLES)}

    _BASE_URL = "https://physionet.org/files/challenge-2019/1.0.0/"
    _SETS = ["training_setA", "training_setB"]

    def download(self) -> None:
        dest = Path(self.root) / "physionet2019"
        if (dest / "training_setA").exists():
            return
        dest.mkdir(parents=True, exist_ok=True)
        for s in self._SETS:
            zip_url = f"{self._BASE_URL}{s}.zip"
            zip_path = dest / f"{s}.zip"
            print(f"Downloading {zip_url} ...")
            urllib.request.urlretrieve(zip_url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest)
            zip_path.unlink()

    def _load(self) -> None:
        dest = Path(self.root) / "physionet2019"
        all_samples, all_times, all_masks, all_labels = [], [], [], []
        F = len(self.VARIABLES)

        for s in self._SETS:
            set_dir = dest / s
            if not set_dir.exists():
                continue
            for psv_file in sorted(set_dir.glob("*.psv")):
                try:
                    x, t_arr, mask, label = self._parse_patient(psv_file, F)
                except Exception:
                    continue
                all_samples.append(x)
                all_times.append(t_arr)
                all_masks.append(mask)
                all_labels.append(label)

        self.samples = all_samples
        self.times = all_times
        self.masks = all_masks
        self.labels = np.array(all_labels, dtype=np.int64)
        self.num_features = F
        self.num_classes = 2

    def _parse_patient(self, path: Path, F: int):
        """Parse one PSV patient file → (x, t_arr, mask, label)."""
        with open(path) as f:
            header = f.readline().strip().split("|")
            rows = [line.strip().split("|") for line in f if line.strip()]

        # Map column names to indices in the PSV
        col_idx = {name: i for i, name in enumerate(header)}
        iculos_col = col_idx.get("ICULOS", -1)
        sepsis_col = col_idx.get("SepsisLabel", -1)

        T = len(rows)
        x = np.zeros((T, F), dtype=np.float32)
        mask = np.zeros((T, F), dtype=np.float32)
        t_arr = np.zeros(T, dtype=np.float32)
        label = 0

        for t_idx, row in enumerate(rows):
            # Time from ICULOS (hours)
            if iculos_col >= 0:
                try:
                    t_arr[t_idx] = float(row[iculos_col])
                except (ValueError, IndexError):
                    t_arr[t_idx] = float(t_idx)
            else:
                t_arr[t_idx] = float(t_idx)

            # SepsisLabel — use last row's label
            if sepsis_col >= 0:
                try:
                    lbl = int(float(row[sepsis_col]))
                    if lbl == 1:
                        label = 1
                except (ValueError, IndexError):
                    pass

            # Feature values
            for var_name, var_i in self._VAR_IDX.items():
                col = col_idx.get(var_name, -1)
                if col < 0 or col >= len(row):
                    continue
                val_str = row[col]
                if val_str not in ("NaN", "nan", "", "NA"):
                    try:
                        x[t_idx, var_i] = float(val_str)
                        mask[t_idx, var_i] = 1.0
                    except ValueError:
                        pass

        return x, t_arr, mask, label
```

- [ ] **Step 4: Update `dataset/irregular/__init__.py`** to add PhysioNet2019

```python
# torch_timeseries/dataset/irregular/__init__.py
from .base import IrregularTimeSeriesDataset
from .physionet2012 import PhysioNet2012
from .physionet2019 import PhysioNet2019

__all__ = ["IrregularTimeSeriesDataset", "PhysioNet2012", "PhysioNet2019"]
```

- [ ] **Step 5: Run all dataset tests**

```bash
pytest tests/dataset/test_irregular_datasets.py -v
```
Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add torch_timeseries/dataset/irregular/ tests/dataset/test_irregular_datasets.py
git commit -m "feat: add PhysioNet2019 dataset"
```

---

### Task 5: `IrregularClassificationDataModule` + Config

**Files:**
- Create: `torch_timeseries/dataloader/v2/irregular_classification.py`
- Test: `tests/dataloader/test_v2_irregular.py` (add new tests)

The DataModule:
1. Splits indices into train/val/test using `SplitConfig` proportions.
2. Fits `scaler` on concatenated observed values of training samples only.
3. Returns `IrregularTSBatch` from each loader with `y` as a class label scalar.

- [ ] **Step 1: Add failing tests**

```python
# append to tests/dataloader/test_v2_irregular.py

import numpy as np
import pytest
import torch


class _ToyIrregular:
    """In-memory irregular dataset — no file I/O."""
    num_features = 3
    num_classes = 2

    def __init__(self, n=40, seed=0):
        rng = np.random.default_rng(seed)
        self.samples, self.times, self.masks, self.labels = [], [], [], []
        for i in range(n):
            T = rng.integers(5, 15)
            x = rng.normal(size=(T, self.num_features)).astype(np.float32)
            t = np.sort(rng.uniform(0, 48, size=T)).astype(np.float32)
            mask = (rng.random((T, self.num_features)) > 0.3).astype(np.float32)
            self.samples.append(x)
            self.times.append(t)
            self.masks.append(mask)
            self.labels.append(i % self.num_classes)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.samples)


def _toy_irreg_cls_dm(**kw):
    from torch_timeseries.dataloader.v2.irregular_classification import (
        IrregularClassificationDataModule, IrregularClassificationConfig,
    )
    from torch_timeseries.dataloader.v2.forecast import SplitConfig, LoaderConfig
    from torch_timeseries.scaler import StandardScaler
    return IrregularClassificationDataModule(
        dataset=_ToyIrregular(n=40),
        scaler=StandardScaler(),
        window=IrregularClassificationConfig(**kw),
        split=SplitConfig(train=0.7, val=0.1, test=0.2),
        loader=LoaderConfig(batch_size=8, num_workers=0),
    )


def test_irregular_classification_dm_returns_batch():
    dm = _toy_irreg_cls_dm()
    batch = next(iter(dm.train_loader))
    from torch_timeseries.dataloader.v2.irregular_batch import IrregularTSBatch
    assert isinstance(batch, IrregularTSBatch)


def test_irregular_classification_dm_batch_shapes():
    dm = _toy_irreg_cls_dm()
    batch = next(iter(dm.train_loader))
    B = batch.x.shape[0]
    T = batch.x.shape[1]
    assert batch.x.shape == (B, T, 3)
    assert batch.t.shape == (B, T)
    assert batch.mask.shape == (B, T, 3)
    assert batch.y.shape == (B,)
    assert batch.y.dtype == torch.long


def test_irregular_classification_dm_t_normalized():
    """All real (non-padded) t values should be in [0, 1]."""
    dm = _toy_irreg_cls_dm()
    batch = next(iter(dm.train_loader))
    # padded positions have t=1.0, real positions have mask>0 somewhere
    # All values should be in [0,1]
    assert batch.t.min() >= 0.0
    assert batch.t.max() <= 1.0


def test_irregular_classification_dm_properties():
    dm = _toy_irreg_cls_dm()
    assert dm.num_features == 3
    assert dm.num_classes == 2
    assert hasattr(dm, "train_loader")
    assert hasattr(dm, "val_loader")
    assert hasattr(dm, "test_loader")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/dataloader/test_v2_irregular.py::test_irregular_classification_dm_returns_batch -v
```
Expected: `ImportError` — module does not exist yet.

- [ ] **Step 3: Write `irregular_classification.py`**

```python
# torch_timeseries/dataloader/v2/irregular_classification.py
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
class IrregularClassificationConfig:
    time_enc: int = 0           # 0=no calendar features, 1=fourier (future)
    freq: Optional[str] = None  # required if time_enc > 0


class _IrregularClassificationDataset(Dataset):
    """Maps indices into an IrregularTimeSeriesDataset, scales, normalizes time."""

    def __init__(self, dataset, scaler, indices: List[int]) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> IrregularTSBatch:
        i = self.indices[idx]
        x_raw = np.array(self.dataset.samples[i], dtype=np.float32)  # (T_i, F)
        t_raw = np.array(self.dataset.times[i], dtype=np.float32)     # (T_i,)
        mask = np.array(self.dataset.masks[i], dtype=np.float32)       # (T_i, F)
        label = int(self.dataset.labels[i])

        # Normalize time to [0, 1] per sample
        t_min, t_max = t_raw.min(), t_raw.max()
        t_norm = (t_raw - t_min) / (t_max - t_min + 1e-8)

        # Scale observed values
        x_scaled = self.scaler.transform(x_raw)

        # Zero-out unobserved positions after scaling to avoid polluting means
        x_scaled = x_scaled * mask

        return IrregularTSBatch(
            x=torch.from_numpy(x_scaled),
            t=torch.from_numpy(t_norm),
            mask=torch.from_numpy(mask),
            y=torch.tensor(label, dtype=torch.long),
        )


class IrregularClassificationDataModule:
    """DataModule for irregular time-series classification.

    Splits by sample index; fits scaler on training set observed values only.
    Each DataLoader returns ``IrregularTSBatch`` via ``collate_irregular``.
    """

    def __init__(
        self,
        dataset,
        scaler,
        window: IrregularClassificationConfig = None,
        split: SplitConfig = None,
        loader: LoaderConfig = None,
    ) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.window_cfg = window or IrregularClassificationConfig()
        self.split_cfg = split or SplitConfig()
        self.loader_cfg = loader or LoaderConfig()

        self._fit_scaler()
        self._build_datasets()
        self._build_loaders()

    def _fit_scaler(self) -> None:
        n = len(self.dataset)
        train_end = int(self.split_cfg.train * n)
        train_indices = list(range(0, train_end))

        obs_list = []
        for i in train_indices:
            x = np.array(self.dataset.samples[i], dtype=np.float32)
            obs_list.append(x)

        if obs_list:
            X_all = np.vstack(obs_list)  # (sum_T, F)
            self.scaler.fit(X_all)

    def _build_datasets(self) -> None:
        n = len(self.dataset)
        train_end = int(self.split_cfg.train * n)
        test_size = int((self.split_cfg.test or 0.2) * n)
        val_end = n - test_size

        train_idx = list(range(0, train_end))
        val_idx = list(range(train_end, val_end))
        test_idx = list(range(val_end, n))

        self.train_dataset = _IrregularClassificationDataset(
            self.dataset, self.scaler, train_idx)
        self.val_dataset = _IrregularClassificationDataset(
            self.dataset, self.scaler, val_idx)
        self.test_dataset = _IrregularClassificationDataset(
            self.dataset, self.scaler, test_idx)

    def _build_loaders(self) -> None:
        lc = self.loader_cfg
        kw = dict(
            batch_size=lc.batch_size,
            num_workers=lc.num_workers,
            pin_memory=lc.pin_memory,
            collate_fn=collate_irregular,
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
    def num_classes(self) -> int:
        return self.dataset.num_classes
```

- [ ] **Step 4: Run the new tests**

```bash
pytest tests/dataloader/test_v2_irregular.py -v
```
Expected: `6 passed` (2 from Task 1 + 4 new)

- [ ] **Step 5: Commit**

```bash
git add torch_timeseries/dataloader/v2/irregular_classification.py tests/dataloader/test_v2_irregular.py
git commit -m "feat: add IrregularClassificationDataModule"
```

---

### Task 6: GRU-D model

**Files:**
- Create: `torch_timeseries/model/irregular/__init__.py`
- Create: `torch_timeseries/model/irregular/grud.py`
- Test: `tests/model/test_irregular_models.py`

GRU-D (Che et al., 2018): GRU with exponential decay on the input when observations are missing. Each step decays the last observed value and the hidden state by learned rates proportional to elapsed time.

- [ ] **Step 1: Write the failing tests**

```python
# tests/model/test_irregular_models.py
import pytest
import torch


def _make_synthetic_batch(B=4, T=10, F=3, device="cpu"):
    """Create a padded IrregularTSBatch for model testing."""
    x = torch.randn(B, T, F)
    t = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
    mask = (torch.rand(B, T, F) > 0.4).float()
    return x.to(device), t.to(device), mask.to(device)


def test_grud_forward_classification():
    """GRU-D returns (B, num_classes) for classification."""
    from torch_timeseries.model.irregular.grud import GRUD
    B, T, F, C = 4, 10, 3, 2
    model = GRUD(input_size=F, hidden_size=16, output_size=C)
    x, t, mask = _make_synthetic_batch(B, T, F)
    out = model(x, t, mask)
    assert out.shape == (B, C), f"Expected ({B}, {C}), got {out.shape}"


def test_grud_no_nan():
    """GRU-D output has no NaN even with fully-masked inputs."""
    from torch_timeseries.model.irregular.grud import GRUD
    B, T, F, C = 4, 10, 3, 2
    model = GRUD(input_size=F, hidden_size=16, output_size=C)
    x = torch.randn(B, T, F)
    t = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
    mask = torch.zeros(B, T, F)  # all missing
    out = model(x, t, mask)
    assert not torch.isnan(out).any()


def test_grud_gradients_flow():
    """GRU-D gradients flow to all parameters."""
    from torch_timeseries.model.irregular.grud import GRUD
    model = GRUD(input_size=3, hidden_size=16, output_size=2)
    x, t, mask = _make_synthetic_batch(B=2, T=8, F=3)
    logits = model(x, t, mask)
    logits.sum().backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/model/test_irregular_models.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Write `model/irregular/__init__.py`**

```python
# torch_timeseries/model/irregular/__init__.py
from .grud import GRUD
__all__ = ["GRUD"]
```

- [ ] **Step 4: Write `grud.py`**

```python
# torch_timeseries/model/irregular/grud.py
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class GRUDCell(nn.Module):
    """Single GRU-D step with input and hidden-state exponential decay."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # GRU cell: input = [x_imputed, mask] → dim F*2
        self.gru_cell = nn.GRUCell(input_size * 2, hidden_size)
        # Learned decay rates (clamped ≥ 0 during forward)
        self.log_gamma_x = nn.Parameter(torch.zeros(input_size))
        self.log_gamma_h = nn.Parameter(torch.zeros(hidden_size))
        # Global feature mean (learned; substituted when no observation exists yet)
        self.x_mean = nn.Parameter(torch.zeros(input_size))

    def forward(
        self,
        x: Tensor,        # (B, F)   current raw input
        m: Tensor,        # (B, F)   mask: 1=observed
        delta: Tensor,    # (B, F)   time elapsed since last obs per feature
        x_last: Tensor,   # (B, F)   last observed value per feature
        h: Tensor,        # (B, H)   previous hidden state
    ) -> Tuple[Tensor, Tensor]:
        gamma_x = torch.exp(-torch.relu(self.log_gamma_x) * delta)          # (B, F)
        x_imputed = m * x + (1.0 - m) * (gamma_x * x_last + (1.0 - gamma_x) * self.x_mean)

        # Per-feature delta → mean across features for hidden decay
        delta_h = delta.mean(dim=-1, keepdim=True).expand_as(h)             # (B, H)
        gamma_h = torch.exp(-torch.relu(self.log_gamma_h) * delta_h)        # (B, H)
        h_decayed = gamma_h * h

        gru_in = torch.cat([x_imputed, m], dim=-1)                          # (B, F*2)
        h_new = self.gru_cell(gru_in, h_decayed)

        # Update x_last only at observed positions
        x_last_new = m * x + (1.0 - m) * x_last
        return h_new, x_last_new


class GRUD(nn.Module):
    """GRU-D: Recurrent Neural Networks for Multivariate Time Series
    with Missing Values (Che et al., 2018).

    forward() returns ``(B, output_size)`` logits (classification).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cell = GRUDCell(input_size, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: Tensor,              # (B, T, F)
        t: Tensor,              # (B, T)
        mask: Tensor,           # (B, T, F)
        x_time: Tensor = None,  # ignored in Phase 1
    ) -> Tensor:                # (B, output_size)
        B, T, F = x.shape
        h = x.new_zeros(B, self.cell.hidden_size)
        x_last = x.new_zeros(B, F)
        t_last = x.new_zeros(B, F)      # last obs time per feature

        for step in range(T):
            x_t = x[:, step, :]                              # (B, F)
            m_t = mask[:, step, :]                           # (B, F)
            t_t = t[:, step].unsqueeze(-1).expand(B, F)     # (B, F)

            delta = torch.clamp(t_t - t_last, min=0.0)      # (B, F)
            h, x_last = self.cell(x_t, m_t, delta, x_last, h)

            # Update t_last only for observed features
            t_last = m_t * t_t + (1.0 - m_t) * t_last

        return self.fc(self.drop(h))
```

- [ ] **Step 5: Run model tests**

```bash
pytest tests/model/test_irregular_models.py -v
```
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add torch_timeseries/model/irregular/ tests/model/test_irregular_models.py
git commit -m "feat: add GRU-D model for irregular time series"
```

---

### Task 7: `IrregularClassificationExp` + `GRUDIrregularClassification` combo

**Files:**
- Create: `torch_timeseries/experiments/irregular_classification.py`
- Create: `torch_timeseries/experiments/GRUD.py`
- Test: `tests/experiments/test_irregular_experiments.py`

`IrregularClassificationExp` follows the same pattern as `UEAClassificationExp`:
- `_init_data_loader()` builds `IrregularClassificationDataModule`
- `_process_one_batch(batch)` sends batch to device and calls `self.model`
- `_train()` / `_evaluate()` iterate `IrregularTSBatch` loaders
- `run(seed)` returns `Dict[str, float]` — `{"accuracy": ..., "cross_entropy": ...}`

The test uses `_ToyIrregular` (no network) and checks that `run(seed=1)` returns a result dict.

- [ ] **Step 1: Write the failing test**

```python
# tests/experiments/test_irregular_experiments.py
import pytest


def _make_toy_grud_exp():
    """Build a GRUDIrregularClassification on _ToyIrregular, no files needed."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../dataloader"))

    from torch_timeseries.experiments.GRUD import GRUDIrregularClassification
    # monkey-patch dataset_type to _ToyIrregular
    return GRUDIrregularClassification(
        dataset_type="__toy__",
        epochs=2,
        patience=5,
        batch_size=8,
        hidden_size=16,
        device="cpu",
        save_dir="/tmp/test_grud_irreg",
    )


def test_grud_irregular_classification_single_run(tmp_path):
    """GRUDIrregularClassification.run() returns a metrics dict."""
    from torch_timeseries.experiments.GRUD import GRUDIrregularClassification
    from tests.dataloader.test_v2_irregular import _ToyIrregular

    exp = GRUDIrregularClassification(
        dataset_type="__toy__",
        epochs=2,
        patience=5,
        batch_size=8,
        hidden_size=16,
        device="cpu",
        save_dir=str(tmp_path),
    )
    # Inject toy dataset directly to avoid file I/O
    exp._toy_dataset = _ToyIrregular(n=40)

    result = exp.run(seed=1)

    assert isinstance(result, dict)
    assert "accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0


def test_grud_registered_in_experiment_builder(tmp_path):
    """Experiment(model='GRUD', task='IrregularClassification', ...) resolves without error."""
    from torch_timeseries.experiments import get_experiment_class
    cls = get_experiment_class("GRUD", "IrregularClassification")
    assert cls is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/experiments/test_irregular_experiments.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Write `irregular_classification.py`**

```python
# torch_timeseries/experiments/irregular_classification.py
from __future__ import annotations

import datetime
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy, MetricCollection
from tqdm import tqdm

from ..core import BaseIrrelevant, BaseRelevant
from ..dataloader.v2.forecast import LoaderConfig, SplitConfig
from ..dataloader.v2.irregular_classification import (
    IrregularClassificationConfig, IrregularClassificationDataModule,
)
from ..dataloader.v2.irregular_batch import IrregularTSBatch
from ..scaler import StandardScaler
from ..utils.early_stop import EarlyStopping
from ..utils.model_stats import count_parameters
from ..utils.reproduce import get_rng_state, reproducible, set_rng_state

_DATASET_REGISTRY: Dict[str, type] = {}


def _register_dataset(name: str, cls: type) -> None:
    _DATASET_REGISTRY[name] = cls


def _get_dataset(name: str, root: str):
    if name in _DATASET_REGISTRY:
        return _DATASET_REGISTRY[name]
    from ..dataset.irregular import PhysioNet2012, PhysioNet2019
    _map = {"PhysioNet2012": PhysioNet2012, "PhysioNet2019": PhysioNet2019}
    if name not in _map:
        raise ValueError(
            f"Unknown irregular classification dataset: {name!r}. "
            f"Available: {list(_map)}"
        )
    return _map[name](root=root)


@dataclass
class IrregularClassificationSettings:
    time_enc: int = 0
    freq: Optional[str] = None


@dataclass
class IrregularClassificationExp(BaseRelevant, BaseIrrelevant, IrregularClassificationSettings):
    """Base experiment for irregular time-series classification."""

    loss_func_type: str = "cross_entropy"

    # --- sub-classes must implement this ---
    def _init_model(self) -> None:
        raise NotImplementedError

    # --------------------------------------------------------------- setup

    def _init_data_loader(self) -> None:
        if hasattr(self, "_toy_dataset"):
            dataset = self._toy_dataset
        else:
            dataset = _get_dataset(self.dataset_type, self.data_path)
        self.dm = IrregularClassificationDataModule(
            dataset=dataset,
            scaler=StandardScaler(),
            window=IrregularClassificationConfig(
                time_enc=self.time_enc, freq=self.freq),
            split=SplitConfig(train=0.7, val=0.1, test=0.2),
            loader=LoaderConfig(
                batch_size=self.batch_size,
                num_workers=self.num_worker,
            ),
        )
        self.train_loader = self.dm.train_loader
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader

    def _init_metrics(self) -> None:
        self.metrics = MetricCollection(
            {"accuracy": Accuracy("multiclass", num_classes=self.dm.num_classes)}
        )
        self.metrics.to(self.device)

    def _init_loss_func(self) -> None:
        self.loss_func = CrossEntropyLoss()

    def _init_optimizer(self) -> None:
        from torch.optim import Adam
        self.optimizer = Adam(
            self.model.parameters(), lr=self.lr,
            weight_decay=self.l2_weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs,
        )

    def _setup(self) -> None:
        self._init_data_loader()
        self._init_metrics()
        self._init_loss_func()
        self.current_epochs = 0
        self.setuped = True

    # --------------------------------------------------------------- batch

    def _process_one_batch(
        self, batch: IrregularTSBatch
    ):
        x = batch.x.float().to(self.device)
        t = batch.t.float().to(self.device)
        mask = batch.mask.float().to(self.device)
        y = batch.y.to(self.device)
        logits = self.model(x, t, mask)
        return logits, y

    # --------------------------------------------------------------- loops

    def _train(self) -> List[float]:
        self.model.train()
        losses = []
        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset), leave=False) as pb:
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                logits, y = self._process_one_batch(batch)
                loss = self.loss_func(logits, y.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)
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
                logits, y = self._process_one_batch(batch)
                loss = self.loss_func(logits, y.long())
                losses.append(loss.item())
                self.metrics.update(logits, y)
        result = {k: float(v.compute()) for k, v in self.metrics.items()}
        result["cross_entropy"] = float(np.mean(losses))
        return result

    # --------------------------------------------------------------- run

    def _run_identifier(self, seed: int) -> str:
        ident = asdict(self)
        ident["seed"] = seed
        return hashlib.md5(
            json.dumps(ident, sort_keys=True, default=str).encode()
        ).hexdigest()

    def _setup_run(self, seed: int) -> None:
        if not hasattr(self, "setuped"):
            self._setup()
        reproducible(seed)
        self._init_model()
        self._init_optimizer()
        self.current_epoch = 0
        self.run_save_dir = os.path.join(
            self.save_dir, "runs", self.model_type,
            str(self.dataset_type), self._run_identifier(seed),
        )
        os.makedirs(self.run_save_dir, exist_ok=True)
        self.best_ckpt = os.path.join(self.run_save_dir, "best_model.pth")
        self.run_ckpt = os.path.join(self.run_save_dir, "run_checkpoint.pth")
        self.early_stopper = EarlyStopping(
            self.patience, verbose=False, path=self.best_ckpt)

    def run(self, seed: int = 42) -> Dict[str, float]:
        self._setup_run(seed)

        while self.current_epoch < self.epochs:
            if self.early_stopper.early_stop:
                break
            reproducible(seed + self.current_epoch)
            self._train()
            val_result = self._evaluate(self.val_loader)
            self.current_epoch += 1
            self.early_stopper(val_result["cross_entropy"], model=self.model)
            self.scheduler.step()

        # Load best and evaluate on test
        if os.path.exists(self.best_ckpt):
            self.model.load_state_dict(
                torch.load(self.best_ckpt, map_location=self.device, weights_only=False)
            )
        return self._evaluate(self.test_loader)

    def runs(self, seeds: List[int] = None) -> List[Dict[str, float]]:
        seeds = seeds or [1, 2, 3]
        return [self.run(seed=s) for s in seeds]
```

- [ ] **Step 4: Write `GRUD.py`**

```python
# torch_timeseries/experiments/GRUD.py
from dataclasses import dataclass
from ..model.irregular.grud import GRUD
from .irregular_classification import IrregularClassificationExp


@dataclass
class GRUDParameters:
    hidden_size: int = 64
    grud_dropout: float = 0.0


@dataclass
class GRUDIrregularClassification(IrregularClassificationExp, GRUDParameters):
    model_type: str = "GRUD"

    def _init_model(self) -> None:
        self.model = GRUD(
            input_size=self.dm.num_features,
            hidden_size=self.hidden_size,
            output_size=self.dm.num_classes,
            dropout=self.grud_dropout,
        ).to(self.device)
```

- [ ] **Step 5: Run experiment tests**

```bash
pytest tests/experiments/test_irregular_experiments.py -v
```
Expected: `2 passed` (the single-run test will fail until registry wiring in Task 8; the registration test fails too)

Note: `test_grud_irregular_classification_single_run` may need registry wiring — see Task 8.

---

### Task 8: Registry wiring + export wiring + pyproject.toml

**Files:**
- Modify: `torch_timeseries/experiments/__init__.py`
- Modify: `torch_timeseries/dataloader/v2/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `IrregularClassification` to `TASK_SUFFIXES` in `registry.py`**

`torch_timeseries/experiments/registry.py` has a hard-coded tuple. Edit line 4:

```python
# before:
TASK_SUFFIXES = ("Forecast", "Imputation", "UEAClassification", "AnomalyDetection")

# after:
TASK_SUFFIXES = ("Forecast", "Imputation", "UEAClassification", "AnomalyDetection",
                 "IrregularClassification")
```

- [ ] **Step 2: Add GRUD imports to `experiments/__init__.py`**

Open `torch_timeseries/experiments/__init__.py` and add after the existing model imports (before `from .registry import ...`):

```python
from .irregular_classification import IrregularClassificationExp
from .GRUD import GRUDIrregularClassification
```

Verify the registry picks it up:

```bash
python -c "from torch_timeseries.experiments import EXPERIMENT_REGISTRY; print([k for k in EXPERIMENT_REGISTRY if 'Irregular' in str(k)])"
```
Expected output includes: `[('GRUD', 'IrregularClassification')]`

- [ ] **Step 3: Add `IrregularTSBatch` + `collate_irregular` + `IrregularClassificationDataModule` exports to `dataloader/v2/__init__.py`**

Add to the existing imports:
```python
from .irregular_batch import IrregularTSBatch, collate_irregular
from .irregular_classification import IrregularClassificationDataModule, IrregularClassificationConfig
```

Add to `__all__`:
```python
"IrregularTSBatch",
"collate_irregular",
"IrregularClassificationDataModule",
"IrregularClassificationConfig",
```

- [ ] **Step 4: Add `[irregular]` optional deps to `pyproject.toml`**

Open `pyproject.toml`. Add the following to `[project.optional-dependencies]` (or create that section if not present):

```toml
[project.optional-dependencies]
irregular = [
    "torchdiffeq>=0.2.3",
    "torchcde>=0.2.5",
    "torch_geometric>=2.0.0",
]
```

- [ ] **Step 5: Run the full test suite for Phase 1**

```bash
pytest tests/dataloader/test_v2_irregular.py tests/dataset/test_irregular_datasets.py tests/model/test_irregular_models.py tests/experiments/test_irregular_experiments.py -v
```
Expected: all pass (at minimum the non-network tests).

- [ ] **Step 6: Run existing tests to verify no regressions**

```bash
pytest tests/ -v --ignore=tests/experiments/test_autoformer.py -x
```
Expected: all pass.

- [ ] **Step 7: Final commit**

```bash
git add torch_timeseries/experiments/registry.py \
        torch_timeseries/experiments/__init__.py \
        torch_timeseries/experiments/irregular_classification.py \
        torch_timeseries/experiments/GRUD.py \
        torch_timeseries/dataloader/v2/__init__.py \
        pyproject.toml
git commit -m "feat: wire IrregularClassificationExp + GRUD into registry and v2 exports"
```

---

## Self-Review

**Spec coverage check:**
- Section 1 (`IrregularTSBatch` + `collate_irregular`): Task 1 ✅
- Section 2 (PhysioNet2012, PhysioNet2019): Tasks 3–4 ✅
- Section 3 (`IrregularClassificationDataModule`): Task 5 ✅
- Section 4 (GRU-D): Task 6 ✅
- Section 5 (`IrregularClassificationExp` + `GRUDIrregularClassification` + registry): Tasks 7–8 ✅
- Section 6 tests (`test_irregular_tsbatch_collation`, `test_physionet2012_loads`, `test_irregular_classification_dm_returns_batch`, `test_grud_forward_classification`, `test_irregular_classification_exp_single_run`): all covered ✅

**MIMIC / UEAIrregular / IrregularWrapper:** Phase 2 — not in scope here ✅
**mTAN / LatentODE / NeuralCDE / Raindrop:** Phase 3 — not in scope here ✅
**Interpolation / Forecast DataModules:** Phase 2 — not in scope here ✅
