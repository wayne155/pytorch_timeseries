# Experiment Builder + CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `Experiment` builder class that lets researchers run end-to-end experiments in one Python call, compare results across models/datasets, and register custom models with `register_model()`. The `pytexp compare` CLI subcommand reads saved JSON results and prints a table.

**Architecture:** `Experiment` is a builder that wraps the existing `get_experiment_class` registry — it assembles the right experiment class instance, calls `run(seed)`, captures timing and git metadata, and wraps the dict return into a `RunResult` saved by any attached backend. `register_model` inserts a model class into `EXPERIMENT_REGISTRY` so `Experiment(model="MyModel", ...)` works immediately. The CLI delegates to `Experiment.compare()`.

**Tech Stack:** Python dataclasses, existing `ForecastExp` / `DLinearForecast` hierarchy, `get_experiment_class`, `RunResult` / `LocalBackend` from Plan 1 (`torch_timeseries/results/`), `fire` for CLI, `count_parameters` for `num_params`.

**Prerequisite:** Plan 1 (`2026-05-28-foundation-results-and-datamodules.md`) must be complete — `RunResult`, `LocalBackend`, and `_get_git_commit` must exist in `torch_timeseries/results/`.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `torch_timeseries/experiment.py` | `Experiment` builder, `register_model` |
| Modify | `torch_timeseries/__init__.py` | expose `Experiment`, `register_model` |
| Modify | `torch_timeseries/cli/exp.py` | add `compare` subcommand |
| Create | `tests/test_experiment_builder.py` | tests for `Experiment` builder |
| Modify | `tests/test_cli_and_registry.py` | add `test_cli_compare_subcommand` |

---

### Task 6: Experiment builder — single run

**Files:**
- Create: `torch_timeseries/experiment.py`
- Create: `tests/test_experiment_builder.py`

**Key behaviour:** `Experiment(model="DLinear", task="Forecast", dataset="ETTh1").run(seed=1)` returns a `RunResult`. No backend = no file is written. With `.with_local(save_dir)` the result is also saved to JSON.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_experiment_builder.py
import types
import pytest

from torch_timeseries.experiment import Experiment, register_model
from torch_timeseries.results.schema import RunResult


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _make_fast_exp_class():
    """Return a minimal experiment class that skips real training."""
    from dataclasses import dataclass
    from torch_timeseries.experiments.forecast import ForecastExp

    @dataclass
    class _FastExp(ForecastExp):
        dataset_type: str = "ETTh1"
        model_type: str = "DLinear"
        epochs: int = 1
        batch_size: int = 4

        def run(self, seed=1):
            return {"mse": 0.5, "mae": 0.4}

        def _build_model(self):
            import torch.nn as nn
            return nn.Linear(1, 1)

    return _FastExp


# ------------------------------------------------------------------ #
# Tests                                                               #
# ------------------------------------------------------------------ #

def test_experiment_run_returns_run_result(monkeypatch):
    fast_cls = _make_fast_exp_class()
    monkeypatch.setitem(
        __import__("torch_timeseries.experiments", fromlist=["EXPERIMENT_REGISTRY"]).EXPERIMENT_REGISTRY,
        ("_FastModel", "Forecast"),
        fast_cls,
    )
    result = Experiment(model="_FastModel", task="Forecast", dataset="ETTh1").run(seeds=[1])
    assert isinstance(result, list)
    assert all(isinstance(r, RunResult) for r in result)
    assert result[0].model == "_FastModel"
    assert result[0].task == "Forecast"
    assert result[0].dataset == "ETTh1"
    assert result[0].seed == 1
    assert "mse" in result[0].metrics


def test_experiment_run_captures_timestamp(monkeypatch):
    fast_cls = _make_fast_exp_class()
    monkeypatch.setitem(
        __import__("torch_timeseries.experiments", fromlist=["EXPERIMENT_REGISTRY"]).EXPERIMENT_REGISTRY,
        ("_FastModel2", "Forecast"),
        fast_cls,
    )
    result = Experiment(model="_FastModel2", task="Forecast", dataset="ETTh1").run(seeds=[1])
    assert result[0].timestamp != ""


def test_experiment_with_local_saves_json(tmp_path, monkeypatch):
    fast_cls = _make_fast_exp_class()
    monkeypatch.setitem(
        __import__("torch_timeseries.experiments", fromlist=["EXPERIMENT_REGISTRY"]).EXPERIMENT_REGISTRY,
        ("_FastModel3", "Forecast"),
        fast_cls,
    )
    Experiment(model="_FastModel3", task="Forecast", dataset="ETTh1") \
        .with_local(save_dir=str(tmp_path)) \
        .run(seeds=[1])
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1


def test_experiment_set_overrides_setting(monkeypatch):
    fast_cls = _make_fast_exp_class()
    monkeypatch.setitem(
        __import__("torch_timeseries.experiments", fromlist=["EXPERIMENT_REGISTRY"]).EXPERIMENT_REGISTRY,
        ("_FastModel4", "Forecast"),
        fast_cls,
    )
    results = Experiment(model="_FastModel4", task="Forecast", dataset="ETTh1") \
        .set(epochs=2) \
        .run(seeds=[1])
    assert results[0].hparams.get("epochs") == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_experiment_builder.py -v
```

Expected: `ModuleNotFoundError: No module named 'torch_timeseries.experiment'`

- [ ] **Step 3: Create `torch_timeseries/experiment.py`**

```python
from __future__ import annotations

import datetime
import time
from dataclasses import asdict
from typing import Dict, List, Optional

from .results.backends import LocalBackend, ResultBackend, WandbBackend, _get_git_commit
from .results.schema import RunResult


class Experiment:
    """Fluent builder for running and comparing time-series experiments.

    Usage::

        result = (
            Experiment(model="DLinear", task="Forecast", dataset="ETTh1")
            .set(windows=96, pred_len=96)
            .with_local(save_dir="./results")
            .run(seeds=[1, 2, 3])
        )
    """

    def __init__(self, model: str, task: str, dataset: str) -> None:
        self.model = model
        self.task = task
        self.dataset = dataset
        self._overrides: Dict = {}
        self._backends: List[ResultBackend] = []

    # ------------------------------------------------------------------ #
    # Builder methods                                                     #
    # ------------------------------------------------------------------ #

    def set(self, **kwargs) -> "Experiment":
        """Override experiment hyperparameters."""
        self._overrides.update(kwargs)
        return self

    def with_local(self, save_dir: str = "./results") -> "Experiment":
        """Attach a LocalBackend — results saved to ``save_dir/*.json``."""
        self._backends.append(LocalBackend(save_dir=save_dir))
        return self

    def with_wandb(self, project: str, entity: str = None) -> "Experiment":
        """Attach a WandbBackend."""
        self._backends.append(WandbBackend(project=project, entity=entity))
        return self

    # ------------------------------------------------------------------ #
    # Execution                                                           #
    # ------------------------------------------------------------------ #

    def run(self, seeds: List[int] = None) -> List[RunResult]:
        """Run the experiment for each seed, return list of RunResult."""
        from torch_timeseries.experiments import get_experiment_class
        from torch_timeseries.utils.model_stats import count_parameters

        if seeds is None:
            seeds = [42]

        exp_cls = get_experiment_class(self.model, self.task)

        results = []
        for seed in seeds:
            exp = exp_cls(dataset_type=self.dataset, **self._overrides)

            t0 = time.time()
            metrics = exp.run(seed)
            elapsed = time.time() - t0

            try:
                _, num_params = count_parameters(exp.model)
            except Exception:
                num_params = 0

            hparams = {}
            try:
                hparams = {k: v for k, v in asdict(exp).items()
                           if isinstance(v, (int, float, str, bool))}
            except Exception:
                pass

            r = RunResult(
                model=self.model,
                task=self.task,
                dataset=self.dataset,
                seed=seed,
                timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
                hparams=hparams,
                metrics=metrics or {},
                num_params=num_params,
                train_time_sec=round(elapsed, 2),
                git_commit=_get_git_commit(),
            )

            for backend in self._backends:
                backend.save(r)

            results.append(r)

        return results

    # ------------------------------------------------------------------ #
    # Class-level helpers                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def grid(
        cls,
        models: List[str],
        tasks: List[str],
        datasets: List[str],
        seeds: List[int] = None,
        save_dir: str = "./results",
        **shared_kwargs,
    ) -> "_GridRunner":
        """Return a runner for all (model × task × dataset) combinations."""
        return _GridRunner(
            models=models, tasks=tasks, datasets=datasets,
            seeds=seeds or [42], save_dir=save_dir,
            shared_kwargs=shared_kwargs,
        )

    @classmethod
    def compare(
        cls,
        save_dir: str = "./results",
        task: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> None:
        """Print a comparison table from saved results in ``save_dir``."""
        from .results.backends import LocalBackend
        backend = LocalBackend(save_dir=save_dir)
        filters = {}
        if task:
            filters["task"] = task
        if dataset:
            filters["dataset"] = dataset
        results = backend.load_all(**filters)
        _print_comparison_table(results)


class _GridRunner:
    def __init__(self, models, tasks, datasets, seeds, save_dir, shared_kwargs):
        self._models = models
        self._tasks = tasks
        self._datasets = datasets
        self._seeds = seeds
        self._save_dir = save_dir
        self._shared_kwargs = shared_kwargs

    def run(self) -> List[RunResult]:
        all_results = []
        for model in self._models:
            for task in self._tasks:
                for dataset in self._datasets:
                    results = (
                        Experiment(model=model, task=task, dataset=dataset)
                        .set(**self._shared_kwargs)
                        .with_local(save_dir=self._save_dir)
                        .run(seeds=self._seeds)
                    )
                    all_results.extend(results)
        return all_results


def _print_comparison_table(results: List[RunResult]) -> None:
    """Print a grouped comparison table to stdout."""
    if not results:
        print("No results found.")
        return

    # Group by (task, dataset)
    from collections import defaultdict
    import statistics

    groups: dict = defaultdict(list)
    for r in results:
        groups[(r.task, r.dataset)].append(r)

    for (task, dataset), group in sorted(groups.items()):
        print(f"\nTask: {task} | Dataset: {dataset}")
        print("─" * 60)

        # Collect metric names from first result
        metric_keys = list(group[0].metrics.keys())
        header = f"{'Model':<20}" + "".join(f"{k:>20}" for k in metric_keys) + f"{'#params':>12}"
        print(header)

        # Group by model within this (task, dataset)
        by_model: dict = defaultdict(list)
        for r in group:
            by_model[r.model].append(r)

        for model, runs in sorted(by_model.items()):
            row = f"{model:<20}"
            for k in metric_keys:
                vals = [r.metrics[k] for r in runs if k in r.metrics]
                if vals:
                    mean = statistics.mean(vals)
                    if len(vals) > 1:
                        std = statistics.stdev(vals)
                        row += f"{f'{mean:.3f}±{std:.3f}':>20}"
                    else:
                        row += f"{f'{mean:.3f}':>20}"
                else:
                    row += f"{'N/A':>20}"
            params = runs[0].num_params
            row += f"{params:>12,}"
            print(row)


def register_model(model_cls) -> None:
    """Add a model class to the experiment registry for all supported tasks.

    ``model_cls`` must subclass ``BaseExp`` and define ``_build_model``.
    Combo classes are auto-generated for each task suffix found in
    ``EXPERIMENT_REGISTRY``'s existing keys.

    Example::

        from torch_timeseries import register_model
        register_model(MyModel)
        Experiment(model="MyModel", task="Forecast", dataset="ETTh1").run([1])
    """
    from dataclasses import dataclass
    from torch_timeseries.experiments import EXPERIMENT_REGISTRY
    from torch_timeseries.experiments.registry import TASK_SUFFIXES
    from torch_timeseries.experiments.forecast import ForecastExp
    from torch_timeseries.experiments.imputation import ImputationExp
    from torch_timeseries.experiments.anomaly_detection import AnomalyDetectionExp
    from torch_timeseries.experiments.uea_classification import UEAClassificationExp

    task_map = {
        "Forecast": ForecastExp,
        "Imputation": ImputationExp,
        "AnomalyDetection": AnomalyDetectionExp,
        "UEAClassification": UEAClassificationExp,
    }

    model_name = model_cls.__name__
    for task_suffix, task_base in task_map.items():
        combo_name = f"{model_name}{task_suffix}"
        combo_cls = dataclass(type(combo_name, (model_cls, task_base), {}))
        EXPERIMENT_REGISTRY[(model_name, task_suffix)] = combo_cls
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_experiment_builder.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add torch_timeseries/experiment.py tests/test_experiment_builder.py
git commit -m "feat: add Experiment builder with single-run, with_local, and set()"
```

---

### Task 7: register_model test + Experiment.grid() + compare() tests

**Files:**
- Modify: `tests/test_experiment_builder.py`

- [ ] **Step 1: Add failing tests**

```python
# append to tests/test_experiment_builder.py

def test_register_model_makes_it_runnable(monkeypatch):
    """register_model should add model to registry so Experiment can find it."""
    from dataclasses import dataclass
    from torch import nn
    from torch_timeseries.experiments.forecast import ForecastExp
    from torch_timeseries.experiment import register_model, Experiment

    @dataclass
    class _MyNewModel(ForecastExp):
        dataset_type: str = "ETTh1"
        model_type: str = "_MyNewModel"
        epochs: int = 1

        def run(self, seed=1):
            return {"mse": 0.42}

        def _build_model(self):
            return nn.Linear(1, 1)

    register_model(_MyNewModel)

    results = Experiment(model="_MyNewModel", task="Forecast", dataset="ETTh1").run(seeds=[1])
    assert len(results) == 1
    assert results[0].model == "_MyNewModel"


def test_experiment_grid_runs_all_combos(monkeypatch):
    """grid() should run model × task × dataset and return all RunResults."""
    from dataclasses import dataclass
    from torch import nn
    from torch_timeseries.experiments.forecast import ForecastExp
    from torch_timeseries.experiment import Experiment
    from torch_timeseries.experiments import EXPERIMENT_REGISTRY

    @dataclass
    class _GridModel(ForecastExp):
        dataset_type: str = "ETTh1"
        model_type: str = "_GridModel"
        epochs: int = 1

        def run(self, seed=1):
            return {"mse": 0.5}

        def _build_model(self):
            return nn.Linear(1, 1)

    EXPERIMENT_REGISTRY[("_GridModel", "Forecast")] = _GridModel

    results = Experiment.grid(
        models=["_GridModel"],
        tasks=["Forecast"],
        datasets=["ETTh1", "ETTm1"],
        seeds=[1, 2],
    ).run()

    assert len(results) == 4    # 1 model × 1 task × 2 datasets × 2 seeds


def test_experiment_compare_prints_table(tmp_path, monkeypatch, capsys):
    from torch_timeseries.results.backends import LocalBackend
    from torch_timeseries.results.schema import RunResult
    from torch_timeseries.experiment import Experiment

    backend = LocalBackend(save_dir=str(tmp_path))
    for seed in [1, 2]:
        backend.save(RunResult(
            model="DLinear", task="Forecast", dataset="ETTh1", seed=seed,
            timestamp="2026-01-01T00:00:00",
            hparams={}, metrics={"mse": 0.38 + seed * 0.01, "mae": 0.27},
            num_params=22000, train_time_sec=10.0, git_commit="abc",
        ))

    Experiment.compare(save_dir=str(tmp_path), task="Forecast")
    captured = capsys.readouterr().out
    assert "DLinear" in captured
    assert "mse" in captured
    assert "ETTh1" in captured
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_experiment_builder.py::test_register_model_makes_it_runnable tests/test_experiment_builder.py::test_experiment_grid_runs_all_combos tests/test_experiment_builder.py::test_experiment_compare_prints_table -v
```

Expected: failures due to missing functionality (register_model exists but grid/compare may need tuning)

- [ ] **Step 3: Run all experiment builder tests to verify they pass**

```bash
pytest tests/test_experiment_builder.py -v
```

Expected: all PASSED

- [ ] **Step 4: Commit**

```bash
git add tests/test_experiment_builder.py
git commit -m "test: add tests for register_model, grid(), and compare()"
```

---

### Task 8: Expose at top level + CLI compare subcommand

**Files:**
- Modify: `torch_timeseries/__init__.py`
- Modify: `torch_timeseries/cli/exp.py`
- Modify: `tests/test_cli_and_registry.py`

- [ ] **Step 1: Write failing CLI test**

Open `tests/test_cli_and_registry.py` and append:

```python
def test_cli_compare_subcommand(tmp_path):
    """pytexp compare --save_dir <dir> should exit 0 and print something."""
    import subprocess, sys, json

    # write a dummy result
    result_file = tmp_path / "DLinear_Forecast_ETTh1_seed1.json"
    result_file.write_text(json.dumps({
        "model": "DLinear", "task": "Forecast", "dataset": "ETTh1", "seed": 1,
        "timestamp": "2026-01-01T00:00:00", "hparams": {}, "metrics": {"mse": 0.38},
        "num_params": 22000, "train_time_sec": 10.0, "git_commit": "abc",
        "history": None,
    }))

    result = subprocess.run(
        [sys.executable, "-m", "torch_timeseries.cli.exp", "compare",
         "--save_dir", str(tmp_path)],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "DLinear" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_cli_and_registry.py::test_cli_compare_subcommand -v
```

Expected: FAIL — `compare` subcommand not found

- [ ] **Step 3: Update `torch_timeseries/__init__.py`**

Current:

```python
import torch_timeseries.dataset
import torch_timeseries.dataloader
import torch_timeseries.model
from .cli.exp import exp
from .results import RunResult, LocalBackend, WandbBackend
```

Replace with:

```python
import torch_timeseries.dataset
import torch_timeseries.dataloader
import torch_timeseries.model
from .cli.exp import exp
from .results import RunResult, LocalBackend, WandbBackend
from .experiment import Experiment, register_model
```

- [ ] **Step 4: Update `torch_timeseries/cli/exp.py`**

Current `cli/exp.py`:

```python
import sys
import fire
from torch_timeseries.experiments import get_experiment_class


def exp():
    if '--model' not in sys.argv or '--task' not in sys.argv:
        print("Usage: --model [DLinear|Autoformer|...] --task [Forecast|Imputation|UEAClassification|AnomalyDetection]")
        return

    model_index = sys.argv.index('--model') + 1
    task_index = sys.argv.index('--task') + 1

    if model_index >= len(sys.argv) or task_index >= len(sys.argv):
        print("No model or task specified after --model or --task")
        return

    model_name = sys.argv[model_index]
    task_name = sys.argv[task_index]

    for idx in sorted({model_index, model_index - 1, task_index, task_index - 1}, reverse=True):
        sys.argv.pop(idx)

    exp_class = get_experiment_class(model_name, task_name)

    fire.Fire(exp_class)


if __name__ == '__main__':
    exp()
```

Replace with:

```python
import sys
import fire
from torch_timeseries.experiments import get_experiment_class


def _run_experiment():
    if '--model' not in sys.argv or '--task' not in sys.argv:
        print("Usage: --model [DLinear|...] --task [Forecast|Imputation|UEAClassification|AnomalyDetection]")
        return

    model_index = sys.argv.index('--model') + 1
    task_index = sys.argv.index('--task') + 1

    if model_index >= len(sys.argv) or task_index >= len(sys.argv):
        print("No model or task specified after --model or --task")
        return

    model_name = sys.argv[model_index]
    task_name = sys.argv[task_index]

    for idx in sorted({model_index, model_index - 1, task_index, task_index - 1}, reverse=True):
        sys.argv.pop(idx)

    exp_class = get_experiment_class(model_name, task_name)
    fire.Fire(exp_class)


def compare(save_dir: str = "./results", task: str = None, dataset: str = None):
    """Print a comparison table from results saved in save_dir.

    Examples::

        pytexp compare --save_dir ./results --task Forecast
        pytexp compare --save_dir ./results --task Forecast --dataset ETTh1
    """
    from torch_timeseries.experiment import Experiment
    Experiment.compare(save_dir=save_dir, task=task, dataset=dataset)


def exp():
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        sys.argv.pop(1)
        fire.Fire(compare)
    else:
        _run_experiment()


if __name__ == '__main__':
    exp()
```

- [ ] **Step 5: Run all tests**

```bash
pytest tests/ -v --tb=short
```

Expected: all PASSED, no regressions.

- [ ] **Step 6: Commit**

```bash
git add torch_timeseries/__init__.py torch_timeseries/cli/exp.py tests/test_cli_and_registry.py
git commit -m "feat: expose Experiment/register_model at top level and add pytexp compare subcommand"
```
