# DLinear Forecast Architecture Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the first vertical slice, `DLinear + Forecast`, to the canonical architecture: flat `Experiment(...)` configuration, validated Task/Model/Runtime configs, v2 `ForecastDataModule`, named `TSBatch`, `RunResult`, and `DLinearForecast` as a compatibility shim.

**Architecture:** Add small, focused config and engine modules beside the existing code, then route `Experiment(model="DLinear", task="Forecast", ...)` through the new path. Keep the legacy `ForecastExp` implementation available for all other model/task combinations until later migrations.

**Tech Stack:** Python 3.8+, dataclasses, PyTorch, torchmetrics, existing `ForecastDataModule`, existing `DLinear` model, existing `RunResult`/`LocalBackend`.

---

## File Map

| File | Responsibility |
|---|---|
| `torch_timeseries/experiments/configs.py` | New validated `ForecastConfig`, `DLinearConfig`, `RuntimeConfig`, and strict flat-config splitter |
| `torch_timeseries/experiments/engine.py` | New reusable `ForecastEngine` for named `TSBatch` training/eval/checkpoint/result behavior |
| `torch_timeseries/experiments/DLinear.py` | Make `DLinearForecast` a compatibility shim around the new engine path |
| `torch_timeseries/experiment.py` | Accept flat constructor kwargs; route `DLinear/Forecast` through the new engine and keep legacy fallback |
| `tests/experiments/test_dlinear_forecast_slice.py` | New tests for strict config parsing, named batches, result records, local backend, and shim behavior |
| `README.md` | Update examples to constructor-first `Experiment(...)` style |

## Task 1: Formal Forecast/DLinear/Runtime Configuration

**Files:**
- Create: `tests/experiments/test_dlinear_forecast_slice.py`
- Create: `torch_timeseries/experiments/configs.py`

- [ ] **Step 1: Write failing config tests**

Create `tests/experiments/test_dlinear_forecast_slice.py` with:

```python
import pytest


def test_forecast_dlinear_config_split_accepts_flat_experiment_kwargs():
    from torch_timeseries.experiments.configs import split_experiment_config

    task_cfg, model_cfg, runtime_cfg = split_experiment_config(
        model="DLinear",
        task="Forecast",
        kwargs={
            "windows": 96,
            "pred_len": 336,
            "horizon": 1,
            "input_columns": [0, 1],
            "target_columns": [2],
            "individual": True,
            "batch_size": 16,
            "epochs": 2,
            "device": "cpu",
            "save_dir": "./tmp-results",
        },
    )

    assert task_cfg.windows == 96
    assert task_cfg.pred_len == 336
    assert task_cfg.horizon == 1
    assert task_cfg.input_columns == [0, 1]
    assert task_cfg.target_columns == [2]
    assert model_cfg.individual is True
    assert runtime_cfg.batch_size == 16
    assert runtime_cfg.epochs == 2
    assert runtime_cfg.device == "cpu"
    assert runtime_cfg.save_dir == "./tmp-results"


def test_forecast_dlinear_config_rejects_irrelevant_model_kwargs():
    from torch_timeseries.experiments.configs import split_experiment_config

    with pytest.raises(TypeError, match="Unknown or irrelevant configuration keys: d_model"):
        split_experiment_config(
            model="DLinear",
            task="Forecast",
            kwargs={"windows": 96, "pred_len": 96, "d_model": 512},
        )


def test_forecast_config_validates_shape_controls():
    from torch_timeseries.experiments.configs import split_experiment_config

    with pytest.raises(ValueError, match="pred_len must be positive"):
        split_experiment_config(
            model="DLinear",
            task="Forecast",
            kwargs={"windows": 96, "pred_len": 0},
        )
```

- [ ] **Step 2: Run config tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest tests/experiments/test_dlinear_forecast_slice.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'torch_timeseries.experiments.configs'`.

- [ ] **Step 3: Implement config objects and strict splitter**

Create `torch_timeseries/experiments/configs.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple


@dataclass
class ForecastConfig:
    windows: int = 96
    pred_len: int = 96
    horizon: int = 1
    train_ratio: float = 0.7
    val_ratio: Optional[float] = None
    test_ratio: float = 0.2
    time_enc: int = 1
    input_columns: Optional[List[int]] = None
    target_columns: Optional[List[int]] = None
    stride: int = 1
    scale_in_train: bool = True

    def validate(self) -> None:
        if self.windows <= 0:
            raise ValueError("windows must be positive")
        if self.pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if not (0 < self.train_ratio < 1):
            raise ValueError("train_ratio must be between 0 and 1")
        if self.test_ratio is not None and not (0 < self.test_ratio < 1):
            raise ValueError("test_ratio must be between 0 and 1")
        if self.val_ratio is not None and not (0 <= self.val_ratio < 1):
            raise ValueError("val_ratio must be between 0 and 1")


@dataclass
class DLinearConfig:
    individual: bool = False

    def validate(self) -> None:
        if not isinstance(self.individual, bool):
            raise ValueError("individual must be a bool")


@dataclass
class RuntimeConfig:
    data_path: str = "./data"
    save_dir: str = "./results"
    device: str = "cpu"
    scaler_type: str = "StandardScaler"
    optm_type: str = "Adam"
    loss_func_type: str = "mse"
    batch_size: int = 32
    num_worker: int = 0
    lr: float = 0.0001
    l2_weight_decay: float = 0.0
    epochs: int = 20
    patience: int = 5
    max_grad_norm: float = 5.0
    invtrans_loss: bool = False
    pin_memory: bool = False

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_worker < 0:
            raise ValueError("num_worker must be non-negative")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


def _field_names(cls) -> set:
    return {field.name for field in fields(cls)}


def _build(cls, values: Dict):
    names = _field_names(cls)
    return cls(**{key: values.pop(key) for key in list(values) if key in names})


def split_experiment_config(model: str, task: str, kwargs: Dict) -> Tuple[ForecastConfig, DLinearConfig, RuntimeConfig]:
    if (model, task) != ("DLinear", "Forecast"):
        raise NotImplementedError("typed config split is only implemented for DLinear/Forecast")

    remaining = dict(kwargs)
    task_cfg = _build(ForecastConfig, remaining)
    model_cfg = _build(DLinearConfig, remaining)
    runtime_cfg = _build(RuntimeConfig, remaining)

    if remaining:
        unknown = ", ".join(sorted(remaining))
        raise TypeError(f"Unknown or irrelevant configuration keys: {unknown}")

    task_cfg.validate()
    model_cfg.validate()
    runtime_cfg.validate()
    return task_cfg, model_cfg, runtime_cfg
```

- [ ] **Step 4: Run config tests and verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest tests/experiments/test_dlinear_forecast_slice.py -q
```

Expected: PASS for the three config tests.

- [ ] **Step 5: Commit config slice**

Run:

```bash
git add torch_timeseries/experiments/configs.py tests/experiments/test_dlinear_forecast_slice.py
git commit -m "feat: add typed configs for dlinear forecast slice"
```

## Task 2: New DLinear Forecast Engine Path

**Files:**
- Modify: `tests/experiments/test_dlinear_forecast_slice.py`
- Create: `torch_timeseries/experiments/engine.py`

- [ ] **Step 1: Add engine tests using fake data/model hooks**

Append to `tests/experiments/test_dlinear_forecast_slice.py`:

```python

def test_dlinear_forecast_engine_uses_tsbatch_and_returns_metrics(monkeypatch, tmp_path):
    import numpy as np
    import pandas as pd
    import torch
    from torch_timeseries.core import Freq, TimeSeriesDataset
    from torch_timeseries.experiments.configs import DLinearConfig, ForecastConfig, RuntimeConfig
    from torch_timeseries.experiments.engine import DLinearForecastEngine

    class TinyForecastDataset(TimeSeriesDataset):
        name = "TinyForecast"
        num_features = 3
        freq = Freq.hours

        def download(self):
            pass

        def _load(self):
            n = 160
            rng = np.random.default_rng(7)
            self.df = pd.DataFrame(
                {"date": pd.date_range("2020-01-01", periods=n, freq="h"),
                 **{f"c{i}": rng.normal(size=n) for i in range(3)}}
            )
            self.dates = self.df[["date"]]
            self.data = self.df.drop("date", axis=1).values
            self.length = n

    seen = {"batch_type": None}

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return x[:, -4:, :] + self.bias

    class TinyEngine(DLinearForecastEngine):
        def _init_dataset(self):
            self.dataset = TinyForecastDataset(root=str(tmp_path / "data"))

        def _build_model(self):
            self.model = TinyModel().to(self.runtime.device)

        def _process_batch(self, batch):
            seen["batch_type"] = type(batch).__name__
            return super()._process_batch(batch)

    engine = TinyEngine(
        model_name="DLinear",
        dataset_name="TinyForecast",
        task_config=ForecastConfig(windows=8, pred_len=4, train_ratio=0.6, test_ratio=0.2),
        model_config=DLinearConfig(),
        runtime_config=RuntimeConfig(epochs=1, batch_size=8, save_dir=str(tmp_path / "runs")),
    )

    result = engine.run(seed=1)

    assert seen["batch_type"] == "TSBatch"
    assert set(result) == {"mse", "mae"}
    assert engine.datamodule.num_target_features == 3
```

- [ ] **Step 2: Run engine test and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest tests/experiments/test_dlinear_forecast_slice.py::test_dlinear_forecast_engine_uses_tsbatch_and_returns_metrics -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'torch_timeseries.experiments.engine'`.

- [ ] **Step 3: Implement `DLinearForecastEngine`**

Create `torch_timeseries/experiments/engine.py`:

```python
from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Dict

import numpy as np
import torch
from torch.nn import L1Loss, MSELoss
from torch.optim import Adam
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from torch_timeseries.dataloader.v2 import ForecastDataModule, LoaderConfig, SplitConfig, WindowConfig
from torch_timeseries.dataset import *
from torch_timeseries.scaler import *
from torch_timeseries.utils.early_stop import EarlyStopping
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.parse_type import parse_type
from torch_timeseries.utils.reproduce import reproducible
from torch_timeseries.model import DLinear

from .configs import DLinearConfig, ForecastConfig, RuntimeConfig


class DLinearForecastEngine:
    def __init__(self, model_name: str, dataset_name: str, task_config: ForecastConfig,
                 model_config: DLinearConfig, runtime_config: RuntimeConfig):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.task = "Forecast"
        self.task_config = task_config
        self.model_config = model_config
        self.runtime = runtime_config
        self.current_epoch = 0

    def _init_dataset(self):
        self.dataset = parse_type(self.dataset_name, globals())(root=self.runtime.data_path)

    def _init_datamodule(self):
        scaler = parse_type(self.runtime.scaler_type, globals())()
        self.datamodule = ForecastDataModule(
            dataset=self.dataset,
            scaler=scaler,
            window=WindowConfig(
                window=self.task_config.windows,
                horizon=self.task_config.horizon,
                steps=self.task_config.pred_len,
                stride=self.task_config.stride,
                include_raw=True,
                include_time=True,
                time_enc=self.task_config.time_enc,
                freq=self.dataset.freq,
                input_columns=self.task_config.input_columns,
                target_columns=self.task_config.target_columns,
            ),
            split=SplitConfig(
                train=self.task_config.train_ratio,
                val=self.task_config.val_ratio,
                test=self.task_config.test_ratio,
            ),
            loader=LoaderConfig(
                batch_size=self.runtime.batch_size,
                num_workers=self.runtime.num_worker,
                shuffle_train=True,
                pin_memory=self.runtime.pin_memory,
            ),
            scale_in_train=self.task_config.scale_in_train,
        )
        self.scaler = scaler
        self.train_loader = self.datamodule.train_loader
        self.val_loader = self.datamodule.val_loader
        self.test_loader = self.datamodule.test_loader

    def _build_model(self):
        self.model = DLinear(
            seq_len=self.task_config.windows,
            pred_len=self.task_config.pred_len,
            enc_in=self.datamodule.num_features,
            individual=self.model_config.individual,
        ).to(self.runtime.device)

    def _init_runtime(self):
        self.metrics = MetricCollection({"mse": MeanSquaredError(), "mae": MeanAbsoluteError()}).to(self.runtime.device)
        self.loss_func = {"mse": MSELoss, "mae": L1Loss}[self.runtime.loss_func_type]()
        self.optimizer = Adam(self.model.parameters(), lr=self.runtime.lr, weight_decay=self.runtime.l2_weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.runtime.epochs)
        self.run_save_dir = os.path.join(
            self.runtime.save_dir,
            "runs",
            self.model_name,
            self.dataset_name,
            f"w{self.task_config.windows}h{self.task_config.horizon}s{self.task_config.pred_len}",
        )
        os.makedirs(self.run_save_dir, exist_ok=True)
        self.best_checkpoint_filepath = os.path.join(self.run_save_dir, "best_model.pth")
        self.early_stopper = EarlyStopping(self.runtime.patience, verbose=False, path=self.best_checkpoint_filepath)

    def setup(self):
        self._init_dataset()
        self._init_datamodule()
        self._build_model()
        self._init_runtime()

    def _process_batch(self, batch):
        batch = batch.to(self.runtime.device)
        x = batch.x.float()
        y = batch.y.float()
        return self.model(x), y, batch.y_raw

    def _evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        self.metrics.reset()
        with torch.no_grad():
            for batch in loader:
                pred, true, y_raw = self._process_batch(batch)
                if self.runtime.invtrans_loss:
                    pred = self.scaler.inverse_transform(pred)
                    true = y_raw.to(self.runtime.device).float()
                self.metrics.update(pred.contiguous(), true.contiguous())
        return {name: float(metric.compute()) for name, metric in self.metrics.items()}

    def _train_epoch(self):
        self.model.train()
        losses = []
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            pred, true, y_raw = self._process_batch(batch)
            if self.runtime.invtrans_loss:
                pred = self.scaler.inverse_transform(pred)
                true = y_raw.to(self.runtime.device).float()
            loss = self.loss_func(pred, true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.runtime.max_grad_norm)
            self.optimizer.step()
            losses.append(loss.item())
        return losses

    def run(self, seed: int = 42) -> Dict[str, float]:
        reproducible(seed)
        self.setup()
        for epoch in range(self.runtime.epochs):
            self.current_epoch = epoch
            reproducible(seed + epoch)
            self._train_epoch()
            val_result = self._evaluate(self.val_loader)
            self.early_stopper(val_result[self.runtime.loss_func_type], model=self.model)
            self.scheduler.step()
            if self.early_stopper.early_stop:
                break
        if os.path.exists(self.best_checkpoint_filepath):
            self.model.load_state_dict(torch.load(self.best_checkpoint_filepath, map_location=self.runtime.device, weights_only=False))
        return self._evaluate(self.test_loader)

    def hparams(self) -> dict:
        out = {}
        out.update(asdict(self.task_config))
        out.update(asdict(self.model_config))
        out.update(asdict(self.runtime))
        return out

    def num_parameters(self) -> int:
        try:
            _, num_params = count_parameters(self.model)
            return num_params
        except Exception:
            return 0
```

- [ ] **Step 4: Run engine test and verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest tests/experiments/test_dlinear_forecast_slice.py::test_dlinear_forecast_engine_uses_tsbatch_and_returns_metrics -q
```

Expected: PASS.

- [ ] **Step 5: Run all slice tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest tests/experiments/test_dlinear_forecast_slice.py -q
```

Expected: all tests in the file pass.

- [ ] **Step 6: Commit engine slice**

Run:

```bash
git add torch_timeseries/experiments/engine.py tests/experiments/test_dlinear_forecast_slice.py
git commit -m "feat: add dlinear forecast engine path"
```

## Task 3: Constructor-First `Experiment(...)` API for DLinear Forecast

**Files:**
- Modify: `tests/experiments/test_dlinear_forecast_slice.py`
- Modify: `torch_timeseries/experiment.py`

- [ ] **Step 1: Add Experiment API tests**

Append to `tests/experiments/test_dlinear_forecast_slice.py`:

```python

def test_experiment_constructor_accepts_flat_dlinear_forecast_config(monkeypatch, tmp_path):
    from torch_timeseries.experiment import Experiment

    captured = {}

    class FakeEngine:
        def __init__(self, model_name, dataset_name, task_config, model_config, runtime_config):
            captured["model_name"] = model_name
            captured["dataset_name"] = dataset_name
            captured["task_config"] = task_config
            captured["model_config"] = model_config
            captured["runtime_config"] = runtime_config
            self.model = None

        def run(self, seed):
            captured["seed"] = seed
            return {"mse": 0.12, "mae": 0.08}

        def hparams(self):
            return {"windows": captured["task_config"].windows, "individual": captured["model_config"].individual}

        def num_parameters(self):
            return 5

    monkeypatch.setattr("torch_timeseries.experiment.DLinearForecastEngine", FakeEngine)

    results = Experiment(
        model="DLinear",
        task="Forecast",
        dataset="ETTh1",
        windows=24,
        pred_len=12,
        individual=True,
        epochs=1,
        save_dir=str(tmp_path),
    ).run(seeds=[7])

    assert captured["model_name"] == "DLinear"
    assert captured["dataset_name"] == "ETTh1"
    assert captured["task_config"].windows == 24
    assert captured["task_config"].pred_len == 12
    assert captured["model_config"].individual is True
    assert captured["runtime_config"].epochs == 1
    assert captured["seed"] == 7
    assert results[0].metrics == {"mse": 0.12, "mae": 0.08}
    assert results[0].hparams == {"windows": 24, "individual": True}
    assert results[0].num_params == 5


def test_experiment_constructor_saves_dlinear_forecast_local_result(monkeypatch, tmp_path):
    from torch_timeseries.experiment import Experiment

    class FakeEngine:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, seed):
            return {"mse": 0.2}

        def hparams(self):
            return {"windows": 8}

        def num_parameters(self):
            return 9

    monkeypatch.setattr("torch_timeseries.experiment.DLinearForecastEngine", FakeEngine)

    results = Experiment(
        model="DLinear",
        task="Forecast",
        dataset="ETTh1",
        windows=8,
        pred_len=4,
        save_dir=str(tmp_path),
    ).run(seeds=[1])

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert results[0].metrics["mse"] == 0.2
```

- [ ] **Step 2: Run API tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest tests/experiments/test_dlinear_forecast_slice.py::test_experiment_constructor_accepts_flat_dlinear_forecast_config tests/experiments/test_dlinear_forecast_slice.py::test_experiment_constructor_saves_dlinear_forecast_local_result -q
```

Expected: FAIL because `Experiment.__init__` does not accept flat kwargs and does not expose `DLinearForecastEngine` for monkeypatching.

- [ ] **Step 3: Modify `Experiment` to route DLinear/Forecast through typed configs and engine**

In `torch_timeseries/experiment.py`:

1. Add imports near the top:

```python
from .experiments.configs import split_experiment_config
from .experiments.engine import DLinearForecastEngine
```

2. Change `Experiment.__init__` to:

```python
    def __init__(self, model: str, task: str, dataset: str, **kwargs) -> None:
        self.model = model
        self.task = task
        self.dataset = dataset
        self._overrides: Dict = dict(kwargs)
        self._backends: List[ResultBackend] = []
        save_dir = self._overrides.get("save_dir")
        if save_dir is not None:
            self._backends.append(LocalBackend(save_dir=save_dir))
```

3. Add helper methods inside `Experiment`:

```python
    def _uses_engine_path(self) -> bool:
        return (self.model, self.task) == ("DLinear", "Forecast")

    def _run_engine_path(self, seed: int) -> RunResult:
        task_cfg, model_cfg, runtime_cfg = split_experiment_config(
            model=self.model,
            task=self.task,
            kwargs=self._overrides,
        )
        engine = DLinearForecastEngine(
            model_name=self.model,
            dataset_name=self.dataset,
            task_config=task_cfg,
            model_config=model_cfg,
            runtime_config=runtime_cfg,
        )
        t0 = time.time()
        metrics = engine.run(seed)
        elapsed = time.time() - t0
        return RunResult(
            model=self.model,
            task=self.task,
            dataset=self.dataset,
            seed=seed,
            timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
            hparams=engine.hparams(),
            metrics=metrics or {},
            num_params=engine.num_parameters(),
            train_time_sec=round(elapsed, 2),
            git_commit=_get_git_commit(),
        )
```

4. In `Experiment.run`, before legacy class lookup, add:

```python
        if self._uses_engine_path():
            results = []
            for seed in seeds:
                r = self._run_engine_path(seed)
                for backend in self._backends:
                    backend.save(r)
                results.append(r)
            return results
```

Keep the existing legacy fallback unchanged after that block.

- [ ] **Step 4: Run API tests and verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest tests/experiments/test_dlinear_forecast_slice.py::test_experiment_constructor_accepts_flat_dlinear_forecast_config tests/experiments/test_dlinear_forecast_slice.py::test_experiment_constructor_saves_dlinear_forecast_local_result -q
```

Expected: PASS.

- [ ] **Step 5: Run existing Experiment builder tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest tests/test_experiment_builder.py -q
```

Expected: PASS. Existing `.set()` and `.with_local()` behavior remains supported.

- [ ] **Step 6: Commit constructor API slice**

Run:

```bash
git add torch_timeseries/experiment.py tests/experiments/test_dlinear_forecast_slice.py
git commit -m "feat: route dlinear forecast experiment through typed engine"
```

## Task 4: DLinearForecast Compatibility Shim

**Files:**
- Modify: `tests/experiments/test_dlinear_forecast_slice.py`
- Modify: `torch_timeseries/experiments/DLinear.py`

- [ ] **Step 1: Add shim test**

Append to `tests/experiments/test_dlinear_forecast_slice.py`:

```python

def test_dlinear_forecast_class_is_compatibility_shim(monkeypatch):
    from torch_timeseries.experiments.DLinear import DLinearForecast

    captured = {}

    def fake_run_engine(self, seed):
        captured["windows"] = self.windows
        captured["pred_len"] = self.pred_len
        captured["individual"] = self.individual
        captured["seed"] = seed
        return {"mse": 0.33}

    monkeypatch.setattr(DLinearForecast, "_run_engine_compat", fake_run_engine)

    exp = DLinearForecast(windows=16, pred_len=8, individual=True)
    result = exp.run(seed=5)

    assert result == {"mse": 0.33}
    assert captured == {"windows": 16, "pred_len": 8, "individual": True, "seed": 5}
```

- [ ] **Step 2: Run shim test and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest tests/experiments/test_dlinear_forecast_slice.py::test_dlinear_forecast_class_is_compatibility_shim -q
```

Expected: FAIL because `DLinearForecast.run()` still uses the inherited legacy implementation and does not call `_run_engine_compat`.

- [ ] **Step 3: Add compatibility override to `DLinearForecast`**

In `torch_timeseries/experiments/DLinear.py`, replace only the `DLinearForecast` class body with this version. Leave `DLinearUEAClassification`, `DLinearAnomalyDetection`, and `DLinearImputation` unchanged.

```python
@dataclass
class DLinearForecast(ForecastExp, DLinearParameters):
    model_type: str = "DLinear"

    def _run_engine_compat(self, seed):
        from torch_timeseries.experiment import Experiment

        kwargs = {
            "horizon": self.horizon,
            "windows": self.windows,
            "pred_len": self.pred_len,
            "train_ratio": self.train_ratio,
            "test_ratio": self.test_ratio,
            "time_enc": self.time_enc,
            "input_columns": self.input_columns or None,
            "target_columns": self.target_columns or None,
            "individual": self.individual,
            "data_path": self.data_path,
            "save_dir": self.save_dir,
            "device": self.device,
            "num_worker": self.num_worker,
            "scaler_type": self.scaler_type,
            "optm_type": self.optm_type,
            "loss_func_type": self.loss_func_type,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "l2_weight_decay": self.l2_weight_decay,
            "epochs": self.epochs,
            "patience": self.patience,
            "max_grad_norm": self.max_grad_norm,
            "invtrans_loss": self.invtrans_loss,
        }
        result = Experiment(
            model="DLinear",
            task="Forecast",
            dataset=self.dataset_type,
            **kwargs,
        ).run(seeds=[seed])
        return result[0].metrics

    def run(self, seed=42):
        return self._run_engine_compat(seed)

    def runs(self, seeds=None):
        seeds = [1, 2, 3, 4, 5] if seeds is None else seeds
        return [self.run(seed=seed) for seed in seeds]
```

Also remove the now-unused `import torch` at the top of `DLinear.py` only if no other class in the file uses it. In the current file the other DLinear task classes use `torch`, so keep it.

- [ ] **Step 4: Run shim test and verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest tests/experiments/test_dlinear_forecast_slice.py::test_dlinear_forecast_class_is_compatibility_shim -q
```

Expected: PASS.

- [ ] **Step 5: Run CLI dispatch test**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest tests/test_cli_and_registry.py::test_cli_dispatches_registered_experiment -q
```

Expected: PASS. `DLinearForecast` remains importable and dispatchable.

- [ ] **Step 6: Commit shim slice**

Run:

```bash
git add torch_timeseries/experiments/DLinear.py tests/experiments/test_dlinear_forecast_slice.py
git commit -m "refactor: make dlinear forecast a compatibility shim"
```

## Task 5: Update Public Examples and Verify the Slice

**Files:**
- Modify: `README.md`
- Test: existing tests from prior tasks

- [ ] **Step 1: Update README Experiment example to constructor-first style**

In `README.md`, replace the multi-seed example under “Experiment builder (Python API)” with:

```python
# multiple seeds, save results to disk
results = Experiment(
    model="DLinear",
    task="Forecast",
    dataset="ETTh1",
    windows=96,
    pred_len=96,
    lr=0.001,
    save_dir="./results",
).run(seeds=[1, 2, 3])
```

Keep the fluent `.set()`/`.with_local()` API undocumented in the main example, but do not remove code support.

- [ ] **Step 2: Add a small README assertion to tests**

Append to `tests/experiments/test_dlinear_forecast_slice.py`:

```python

def test_readme_teaches_constructor_first_experiment_api():
    text = open("README.md", encoding="utf-8").read()
    assert "Experiment(\n    model=\"DLinear\"" in text
    assert "save_dir=\"./results\"" in text
```

- [ ] **Step 3: Run README assertion and verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest tests/experiments/test_dlinear_forecast_slice.py::test_readme_teaches_constructor_first_experiment_api -q
```

Expected: PASS.

- [ ] **Step 4: Run full focused verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest \
  tests/experiments/test_dlinear_forecast_slice.py \
  tests/test_experiment_builder.py \
  tests/test_cli_and_registry.py \
  tests/experiments/test_experiment_runtime_edges.py \
  tests/dataloader/test_v2_task_modules.py \
  -q
```

Expected: PASS. If `tests/dataloader/test_v2_task_modules.py::test_imputation_dm_empty_val_split` fails due to the known existing imputation split issue, record it as unrelated and still run all other listed tests.

- [ ] **Step 5: Run broader suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest tests/ -q
```

Expected: PASS except any pre-existing failure already documented before this plan. Do not hide failures; report exact failing test names.

- [ ] **Step 6: Commit docs/test update**

Run:

```bash
git add README.md tests/experiments/test_dlinear_forecast_slice.py
git commit -m "docs: teach constructor-first experiment API"
```

## Self-Review Notes

- Spec coverage: This plan implements the agreed first vertical slice only: `DLinear + Forecast`. Other tasks/models remain legacy or existing code paths.
- Public compatibility: `DLinearForecast` remains importable and CLI dispatch still targets the same class name.
- Strict validation: Unknown flat config keys fail for the new engine path.
- Named batches: The new engine consumes `TSBatch` from `ForecastDataModule` directly.
- Results boundary: `Experiment.run()` returns and optionally stores `RunResult`; task/model code does not write leaderboard records.
