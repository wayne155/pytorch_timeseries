# Experiment CLI and Runtime Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make experiment startup and common runtime paths fail predictably, avoid unsafe CLI dispatch, and cover the fixes with lightweight tests that do not download datasets.

**Architecture:** Add a small experiment registry as the single lookup point for `--model`/`--task`, keep the public `pytexp --model ... --task ...` interface compatible, and harden experiment base classes around optional wandb usage and run-state bookkeeping. Tests use tiny fake experiment classes and monkeypatching so they validate behavior without training real models.

**Tech Stack:** Python 3.8+, pytest, Fire CLI dispatch, PyTorch experiment classes, setuptools/pyproject packaging metadata.

---

## Problems To Fix

1. **High: CLI uses `eval()` for experiment dispatch.**  
   File: `torch_timeseries/cli/exp.py:26-32`  
   Reason: `eval(model_exp)` resolves names from the CLI argument string. It is avoidable and makes error handling depend on global imports.  
   Verification: `pytexp --model NotA --task Forecast run 1` should report known experiment combinations without using `eval`.

2. **High: optional wandb paths can crash when wandb is missing or half-configured.**  
   Files: `torch_timeseries/experiments/forecast.py:47-89`, `torch_timeseries/experiments/imputation.py:49-85`, `torch_timeseries/experiments/anomaly_detection.py:49-92`  
   Reason: forecast sets `self.wandb = True` even if module-level `wandb` is `None`; imputation/anomaly import wandb inside `config_wandb` and also reference the wrong property name `result_related_config`.  
   Verification: monkeypatch module `wandb = None`, call `config_wandb(...)`, and assert a clear `RuntimeError` instead of an `AttributeError` or `ModuleNotFoundError`.

3. **Medium: imputation run logs an undefined model parameter attribute.**  
   File: `torch_timeseries/experiments/imputation.py:398-403`  
   Reason: `model_parameters_num` is computed locally, but wandb logging writes `self.model_parameters_num`, which is never set.  
   Verification: enable fake wandb after `_setup_run`, call the parameter logging portion through a small test subclass, and assert it writes the computed integer.

4. **Medium: tests are not reliably discovered and currently prefer slow dataset downloads.**  
   Files: `tests/dataloader/slidingwindowts.py`, `tests/experiments/test_autoformer.py`  
   Reason: `tests/dataloader/slidingwindowts.py` does not match pytest's default `test*.py` pattern, and both existing tests instantiate real datasets or subprocess full training.  
   Verification: `pytest --collect-only tests -q` should collect the new lightweight unit tests.

5. **Medium: dependency metadata is inconsistent across install paths.**  
   Files: `pyproject.toml:15-27`, `setup.py:6-17`, `requirements.txt:1-11`  
   Reason: `pyproject.toml` does not list `torch`, while `setup.py` does; `setup.py` lists `torch_scatter`, while `pyproject.toml` and `requirements.txt` do not; `requirements.txt` uses `torchmetrics>=1.0.0`, while pyproject/setup use `>=1.1.1`. This makes editable/source installs and wheel installs resolve different environments.  
   Verification: a metadata test parses all three files and asserts the shared core runtime dependencies are present and aligned.

## File Structure

- Create: `torch_timeseries/experiments/registry.py`  
  Responsibility: provide `EXPERIMENT_REGISTRY`, `get_experiment_class(model, task)`, and `available_experiments()` with no CLI parsing concerns.

- Modify: `torch_timeseries/experiments/__init__.py`  
  Responsibility: keep existing public imports, export registry helpers, and avoid duplicating lookup logic in CLI.

- Modify: `torch_timeseries/cli/exp.py`  
  Responsibility: parse `--model`/`--task`, remove those args from `sys.argv`, resolve the experiment class through the registry, and pass the class to Fire.

- Create: `tests/experiments/test_registry.py`  
  Responsibility: prove known experiment classes are registered and unknown combinations produce useful errors.

- Create: `tests/cli/test_exp_cli.py`  
  Responsibility: prove CLI argv mutation and Fire dispatch behavior without launching training.

- Create: `tests/experiments/test_optional_wandb.py`  
  Responsibility: prove wandb configuration fails clearly when unavailable and uses `result_related_configs` consistently.

- Create: `tests/experiments/test_imputation_runtime.py`  
  Responsibility: prove imputation parameter logging uses the computed local parameter count.

- Create: `tests/test_dependency_metadata.py`  
  Responsibility: prove dependency declarations stay aligned across `pyproject.toml`, `setup.py`, and `requirements.txt`.

- Modify: `torch_timeseries/experiments/forecast.py`  
  Responsibility: make `config_wandb()` reject missing wandb clearly before setting `self.wandb`; keep existing public method name.

- Modify: `torch_timeseries/experiments/imputation.py`  
  Responsibility: same wandb availability handling, fix `result_related_config` to `result_related_configs`, and fix parameter summary logging.

- Modify: `torch_timeseries/experiments/anomaly_detection.py`  
  Responsibility: same wandb availability handling and fix `result_related_config` to `result_related_configs`.

- Modify: `pyproject.toml`, `setup.py`, `requirements.txt`  
  Responsibility: align the core dependency list so the documented install and package install paths agree.

---

### Task 1: Add Experiment Registry

**Files:**
- Create: `torch_timeseries/experiments/registry.py`
- Modify: `torch_timeseries/experiments/__init__.py`
- Test: `tests/experiments/test_registry.py`

- [ ] **Step 1: Write the failing registry tests**

Create `tests/experiments/test_registry.py`:

```python
import pytest

from torch_timeseries.experiments import AutoformerForecast, DLinearImputation
from torch_timeseries.experiments.registry import (
    available_experiments,
    get_experiment_class,
)


def test_get_experiment_class_returns_known_experiment():
    assert get_experiment_class("Autoformer", "Forecast") is AutoformerForecast
    assert get_experiment_class("DLinear", "Imputation") is DLinearImputation


def test_get_experiment_class_error_lists_available_combinations():
    with pytest.raises(NotImplementedError) as exc:
        get_experiment_class("MissingModel", "Forecast")

    message = str(exc.value)
    assert "Unknown experiment: MissingModelForecast" in message
    assert "AutoformerForecast" in message
    assert "DLinearForecast" in message


def test_available_experiments_is_sorted_and_human_readable():
    values = available_experiments()

    assert values == sorted(values)
    assert "AutoformerForecast" in values
    assert "DLinearImputation" in values
```

- [ ] **Step 2: Run the registry tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest tests/experiments/test_registry.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'torch_timeseries.experiments.registry'`.

- [ ] **Step 3: Implement the minimal registry**

Create `torch_timeseries/experiments/registry.py`:

```python
from typing import Dict, List, Tuple, Type

from .Autoformer import (
    AutoformerAnomalyDetection,
    AutoformerForecast,
    AutoformerImputation,
    AutoformerUEAClassification,
)
from .CATS import CATSForecast
from .Crossformer import (
    CrossformerAnomalyDetection,
    CrossformerForecast,
    CrossformerImputation,
    CrossformerUEAClassification,
)
from .DLinear import (
    DLinearAnomalyDetection,
    DLinearForecast,
    DLinearImputation,
    DLinearUEAClassification,
)
from .FEDformer import (
    FEDformerAnomalyDetection,
    FEDformerForecast,
    FEDformerImputation,
    FEDformerUEAClassification,
)
from .FITS import FITSForecast, FITSUEAClassification
from .FreTS import FreTSForecast, FreTSUEAClassification
from .Informer import (
    InformerAnomalyDetection,
    InformerForecast,
    InformerImputation,
    InformerUEAClassification,
)
from .PatchTST import (
    PatchTSTAnomalyDetection,
    PatchTSTForecast,
    PatchTSTImputation,
    PatchTSTUEAClassification,
)
from .SCINet import (
    SCINetAnomalyDetection,
    SCINetForecast,
    SCINetImputation,
    SCINetUEAClassification,
)
from .TSMixer import (
    TSMixerAnomalyDetection,
    TSMixerForecast,
    TSMixerImputation,
    TSMixerUEAClassification,
)
from .TimesNet import TimesNetForecast, TimesNetUEAClassification
from .iTransformer import (
    iTransformerAnomalyDetection,
    iTransformerForecast,
    iTransformerImputation,
    iTransformerUEAClassification,
)


EXPERIMENT_REGISTRY: Dict[Tuple[str, str], Type] = {
    ("Autoformer", "AnomalyDetection"): AutoformerAnomalyDetection,
    ("Autoformer", "Forecast"): AutoformerForecast,
    ("Autoformer", "Imputation"): AutoformerImputation,
    ("Autoformer", "UEAClassification"): AutoformerUEAClassification,
    ("CATS", "Forecast"): CATSForecast,
    ("Crossformer", "AnomalyDetection"): CrossformerAnomalyDetection,
    ("Crossformer", "Forecast"): CrossformerForecast,
    ("Crossformer", "Imputation"): CrossformerImputation,
    ("Crossformer", "UEAClassification"): CrossformerUEAClassification,
    ("DLinear", "AnomalyDetection"): DLinearAnomalyDetection,
    ("DLinear", "Forecast"): DLinearForecast,
    ("DLinear", "Imputation"): DLinearImputation,
    ("DLinear", "UEAClassification"): DLinearUEAClassification,
    ("FEDformer", "AnomalyDetection"): FEDformerAnomalyDetection,
    ("FEDformer", "Forecast"): FEDformerForecast,
    ("FEDformer", "Imputation"): FEDformerImputation,
    ("FEDformer", "UEAClassification"): FEDformerUEAClassification,
    ("FITS", "Forecast"): FITSForecast,
    ("FITS", "UEAClassification"): FITSUEAClassification,
    ("FreTS", "Forecast"): FreTSForecast,
    ("FreTS", "UEAClassification"): FreTSUEAClassification,
    ("Informer", "AnomalyDetection"): InformerAnomalyDetection,
    ("Informer", "Forecast"): InformerForecast,
    ("Informer", "Imputation"): InformerImputation,
    ("Informer", "UEAClassification"): InformerUEAClassification,
    ("PatchTST", "AnomalyDetection"): PatchTSTAnomalyDetection,
    ("PatchTST", "Forecast"): PatchTSTForecast,
    ("PatchTST", "Imputation"): PatchTSTImputation,
    ("PatchTST", "UEAClassification"): PatchTSTUEAClassification,
    ("SCINet", "AnomalyDetection"): SCINetAnomalyDetection,
    ("SCINet", "Forecast"): SCINetForecast,
    ("SCINet", "Imputation"): SCINetImputation,
    ("SCINet", "UEAClassification"): SCINetUEAClassification,
    ("TSMixer", "AnomalyDetection"): TSMixerAnomalyDetection,
    ("TSMixer", "Forecast"): TSMixerForecast,
    ("TSMixer", "Imputation"): TSMixerImputation,
    ("TSMixer", "UEAClassification"): TSMixerUEAClassification,
    ("TimesNet", "Forecast"): TimesNetForecast,
    ("TimesNet", "UEAClassification"): TimesNetUEAClassification,
    ("iTransformer", "AnomalyDetection"): iTransformerAnomalyDetection,
    ("iTransformer", "Forecast"): iTransformerForecast,
    ("iTransformer", "Imputation"): iTransformerImputation,
    ("iTransformer", "UEAClassification"): iTransformerUEAClassification,
}


def available_experiments() -> List[str]:
    return sorted(f"{model}{task}" for model, task in EXPERIMENT_REGISTRY)


def get_experiment_class(model: str, task: str) -> Type:
    try:
        return EXPERIMENT_REGISTRY[(model, task)]
    except KeyError as exc:
        available = ", ".join(available_experiments())
        raise NotImplementedError(
            f"Unknown experiment: {model}{task}. Available experiments: {available}"
        ) from exc
```

- [ ] **Step 4: Export registry helpers**

Append these imports to `torch_timeseries/experiments/__init__.py` after the existing imports:

```python
from .registry import EXPERIMENT_REGISTRY, available_experiments, get_experiment_class
```

- [ ] **Step 5: Run the registry tests and verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest tests/experiments/test_registry.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add torch_timeseries/experiments/registry.py torch_timeseries/experiments/__init__.py tests/experiments/test_registry.py
git commit -m "test: cover experiment registry"
```

Expected: commit succeeds if the user wants commits during execution. If the working tree has unrelated user changes in these files, skip the commit and report the conflict.

---

### Task 2: Replace CLI eval Dispatch

**Files:**
- Modify: `torch_timeseries/cli/exp.py`
- Test: `tests/cli/test_exp_cli.py`

- [ ] **Step 1: Write the failing CLI tests**

Create `tests/cli/test_exp_cli.py`:

```python
import sys

import pytest

from torch_timeseries.cli import exp as exp_module


class FakeExperiment:
    pass


def test_exp_dispatches_registered_class_and_removes_model_task_args(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pytexp",
            "--model",
            "FakeModel",
            "--task",
            "Forecast",
            "--dataset_type",
            "ETTh1",
            "run",
            "3",
        ],
    )
    monkeypatch.setattr(
        exp_module,
        "get_experiment_class",
        lambda model, task: FakeExperiment,
    )
    monkeypatch.setattr(exp_module.fire, "Fire", lambda cls: captured.setdefault("cls", cls))

    exp_module.exp()

    assert captured["cls"] is FakeExperiment
    assert sys.argv == ["pytexp", "--dataset_type", "ETTh1", "run", "3"]


def test_exp_raises_registry_error_for_unknown_experiment(monkeypatch):
    def raise_unknown(model, task):
        raise NotImplementedError(f"Unknown experiment: {model}{task}")

    monkeypatch.setattr(sys, "argv", ["pytexp", "--model", "Missing", "--task", "Forecast"])
    monkeypatch.setattr(exp_module, "get_experiment_class", raise_unknown)

    with pytest.raises(NotImplementedError, match="Unknown experiment: MissingForecast"):
        exp_module.exp()
```

- [ ] **Step 2: Run the CLI tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest tests/cli/test_exp_cli.py -q`

Expected: FAIL because `torch_timeseries.cli.exp` does not expose `get_experiment_class`, and current implementation uses `eval`.

- [ ] **Step 3: Replace `eval` with registry lookup**

Modify `torch_timeseries/cli/exp.py` to:

```python
import sys

import fire

from torch_timeseries.experiments import get_experiment_class


def exp():
    if "--model" not in sys.argv or "--task" not in sys.argv:
        print(
            "Usage: --model [DLinear|Autoformer|...] --task "
            "[Forecast|Imputation|UEAClassification|AnomalyDetection]"
        )
        return

    model_index = sys.argv.index("--model") + 1
    task_index = sys.argv.index("--task") + 1

    if model_index >= len(sys.argv) or task_index >= len(sys.argv):
        print("No model or task specified after --model or --task")
        return

    model_name = sys.argv[model_index]
    task_name = sys.argv[task_index]

    for idx in sorted(
        {model_index, model_index - 1, task_index, task_index - 1},
        reverse=True,
    ):
        sys.argv.pop(idx)

    exp_class = get_experiment_class(model_name, task_name)
    fire.Fire(exp_class)


if __name__ == "__main__":
    exp()
```

- [ ] **Step 4: Run CLI and registry tests**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest tests/cli/test_exp_cli.py tests/experiments/test_registry.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add torch_timeseries/cli/exp.py tests/cli/test_exp_cli.py
git commit -m "fix: dispatch experiments through registry"
```

Expected: commit succeeds if the user wants commits during execution. If the working tree has unrelated user changes in these files, skip the commit and report the conflict.

---

### Task 3: Harden Optional wandb Configuration

**Files:**
- Modify: `torch_timeseries/experiments/forecast.py`
- Modify: `torch_timeseries/experiments/imputation.py`
- Modify: `torch_timeseries/experiments/anomaly_detection.py`
- Test: `tests/experiments/test_optional_wandb.py`

- [ ] **Step 1: Write the failing wandb availability tests**

Create `tests/experiments/test_optional_wandb.py`:

```python
import pytest

from torch_timeseries.experiments import anomaly_detection, forecast, imputation


def test_forecast_config_wandb_requires_installed_wandb(monkeypatch):
    monkeypatch.setattr(forecast, "wandb", None)
    exp = forecast.ForecastExp(model_type="DLinear", dataset_type="ETTh1")

    with pytest.raises(RuntimeError, match="wandb is not installed"):
        exp.config_wandb("project")

    assert not hasattr(exp, "wandb")


def test_imputation_config_wandb_requires_installed_wandb(monkeypatch):
    monkeypatch.setattr(imputation, "wandb", None)
    exp = imputation.ImputationExp(model_type="DLinear", dataset_type="ETTh1")

    with pytest.raises(RuntimeError, match="wandb is not installed"):
        exp.config_wandb("project", "run-name")

    assert not hasattr(exp, "wandb")


def test_anomaly_config_wandb_requires_installed_wandb(monkeypatch):
    monkeypatch.setattr(anomaly_detection, "wandb", None)
    exp = anomaly_detection.AnomalyDetectionExp(
        model_type="DLinear",
        dataset_type="MSL",
    )

    with pytest.raises(RuntimeError, match="wandb is not installed"):
        exp.config_wandb("project", "run-name")

    assert not hasattr(exp, "wandb")
```

- [ ] **Step 2: Run wandb availability tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest tests/experiments/test_optional_wandb.py -q`

Expected: FAIL. Forecast does not raise, and imputation/anomaly try to import wandb directly or hit the wrong property path.

- [ ] **Step 3: Add a helper to each experiment module**

In `torch_timeseries/experiments/forecast.py`, `torch_timeseries/experiments/imputation.py`, and `torch_timeseries/experiments/anomaly_detection.py`, add this function after the module-level `try/except ImportError`:

```python
def _require_wandb():
    if wandb is None:
        raise RuntimeError(
            "wandb is not installed. Install wandb or run without config_wandb()."
        )
    return wandb
```

- [ ] **Step 4: Use the helper in `ForecastExp.config_wandb`**

Change `ForecastExp.config_wandb` to:

```python
    def config_wandb(
        self,
        project: str,
    ):
        _require_wandb()
        self.project = project
        self.wandb = True
        return self
```

- [ ] **Step 5: Use the helper and correct property name in `ImputationExp.config_wandb`**

Change the start of `ImputationExp.config_wandb` to:

```python
    def config_wandb(
        self,
        project: str,
        name: str,
        mode: str = "online",
    ):
        wandb_client = _require_wandb()

        def convert_dict(dictionary):
            converted_dict = {}
            for key, value in dictionary.items():
                converted_dict[f"config.{key}"] = value
            return converted_dict

        api = wandb_client.Api()
        config_filter = convert_dict(self.result_related_configs)
        runs = api.runs(path=project, filters=config_filter)
```

Then replace every `wandb.` call in that method with `wandb_client.`.

- [ ] **Step 6: Use the helper and correct property name in `AnomalyDetectionExp.config_wandb`**

Change the start of `AnomalyDetectionExp.config_wandb` to:

```python
    def config_wandb(
        self,
        project: str,
        name: str,
        mode: str = "online",
    ):
        wandb_client = _require_wandb()

        def convert_dict(dictionary):
            converted_dict = {}
            for key, value in dictionary.items():
                converted_dict[f"config.{key}"] = value
            return converted_dict

        api = wandb_client.Api()
        config_filter = convert_dict(self.result_related_configs)
        runs = api.runs(path=project, filters=config_filter)
```

Then replace every `wandb.` call in that method with `wandb_client.`. Also change the duplicate-run message to avoid `self.horizon`, because anomaly settings do not define it:

```python
f"{self.model_type} {self.dataset_type} w{self.windows} "
f"ar{self.anomaly_ratio} Experiment already reported, quiting..."
```

- [ ] **Step 7: Run wandb availability tests and verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest tests/experiments/test_optional_wandb.py -q`

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

Run:

```bash
git add torch_timeseries/experiments/forecast.py torch_timeseries/experiments/imputation.py torch_timeseries/experiments/anomaly_detection.py tests/experiments/test_optional_wandb.py
git commit -m "fix: fail clearly when wandb is unavailable"
```

Expected: commit succeeds if the user wants commits during execution. If the working tree has unrelated user changes in these files, skip the commit and report the conflict.

---

### Task 4: Fix Imputation wandb Parameter Summary

**Files:**
- Modify: `torch_timeseries/experiments/imputation.py`
- Test: `tests/experiments/test_imputation_runtime.py`

- [ ] **Step 1: Write the failing imputation summary test**

Create `tests/experiments/test_imputation_runtime.py`:

```python
from types import SimpleNamespace

from torch_timeseries.experiments import imputation


class DummySummary(dict):
    pass


class DummyWandb:
    def __init__(self):
        self.run = SimpleNamespace(summary=DummySummary())

    def log(self, *args, **kwargs):
        pass


def test_imputation_run_logs_computed_parameter_count(monkeypatch):
    dummy_wandb = DummyWandb()
    exp = imputation.ImputationExp(model_type="DLinear", dataset_type="ETTh1")
    exp.wandb = True
    exp.model = object()
    exp.current_epoch = 0
    exp.epochs = 0

    monkeypatch.setattr(imputation, "wandb", dummy_wandb)
    monkeypatch.setattr(imputation, "count_parameters", lambda model: ("table", 123))
    monkeypatch.setattr(exp, "_setup_run", lambda seed: None)
    monkeypatch.setattr(exp, "_check_run_exist", lambda seed: False)
    monkeypatch.setattr(exp, "_run_print", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp, "_load_best_model", lambda: None)
    monkeypatch.setattr(exp, "_test", lambda: {"mse": 0.0})

    result = exp.run(seed=7)

    assert result == {"mse": 0.0}
    assert dummy_wandb.run.summary["parameters"] == 123
```

- [ ] **Step 2: Run the imputation runtime test and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest tests/experiments/test_imputation_runtime.py -q`

Expected: FAIL with `AttributeError: 'ImputationExp' object has no attribute 'model_parameters_num'`.

- [ ] **Step 3: Fix the parameter summary assignment**

In `torch_timeseries/experiments/imputation.py`, change:

```python
        if self._use_wandb():
            wandb.run.summary["parameters"] = self.model_parameters_num
```

to:

```python
        if self._use_wandb():
            wandb.run.summary["parameters"] = model_parameters_num
```

- [ ] **Step 4: Run the imputation runtime test and verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest tests/experiments/test_imputation_runtime.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add torch_timeseries/experiments/imputation.py tests/experiments/test_imputation_runtime.py
git commit -m "fix: log imputation parameter count"
```

Expected: commit succeeds if the user wants commits during execution. If the working tree has unrelated user changes in these files, skip the commit and report the conflict.

---

### Task 5: Align Dependency Metadata

**Files:**
- Modify: `pyproject.toml`
- Modify: `setup.py`
- Modify: `requirements.txt`
- Test: `tests/test_dependency_metadata.py`

- [ ] **Step 1: Write the failing metadata alignment test**

Create `tests/test_dependency_metadata.py`:

```python
import ast
import pathlib

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = pathlib.Path(__file__).resolve().parents[1]


def normalize(requirement):
    return requirement.split(">=")[0].split("==")[0].strip().lower()


def setup_install_requires():
    module = ast.parse((ROOT / "setup.py").read_text())
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "install_requires":
                    return [value.value for value in node.value.elts]
    raise AssertionError("install_requires not found in setup.py")


def test_core_dependency_metadata_is_aligned():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pyproject_deps = {
        normalize(dep) for dep in pyproject["project"]["dependencies"]
    }
    setup_deps = {normalize(dep) for dep in setup_install_requires()}
    requirements_deps = {
        normalize(line)
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    expected = {
        "numpy",
        "pandas",
        "torch",
        "sktime",
        "scikit-learn",
        "tqdm",
        "einops",
        "prettytable",
        "torchmetrics",
        "torchvision",
        "fire",
        "pyyaml",
    }

    assert expected <= pyproject_deps
    assert expected <= setup_deps
    assert expected <= requirements_deps
    assert "torch_scatter" not in setup_deps
```

- [ ] **Step 2: Run metadata test and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest tests/test_dependency_metadata.py -q`

Expected: FAIL because `pyproject.toml` lacks `torch`, `setup.py` lacks `prettytable`, `torchvision`, and `PyYAML`, and `setup.py` includes `torch_scatter`.

- [ ] **Step 3: Align `pyproject.toml` dependencies**

Update `pyproject.toml` dependencies to include `torch` with the existing core dependencies:

```toml
dependencies = [
    "numpy",
    "pandas",
    "torch",
    "sktime>=0.29.0",
    "scikit-learn",
    "tqdm",
    "einops",
    "prettytable",
    "torchmetrics>=1.1.1",
    "torchvision",
    "fire>=0.5.0",
    "PyYAML",
]
```

- [ ] **Step 4: Align `setup.py` `install_requires`**

Update `setup.py` `install_requires` to:

```python
install_requires = [
    "numpy",
    "pandas",
    "torch",
    "sktime>=0.29.0",
    "scikit-learn",
    "tqdm",
    "einops",
    "prettytable",
    "torchmetrics>=1.1.1",
    "torchvision",
    "fire>=0.5.0",
    "PyYAML",
]
```

Do not include `torch_scatter` unless a later task adds an optional extra with a tested import path.

- [ ] **Step 5: Align `requirements.txt`**

Update `requirements.txt` to:

```text
numpy
pandas
torch
sktime>=0.29.0
scikit-learn
tqdm
einops
prettytable
torchmetrics>=1.1.1
torchvision
fire>=0.5.0
PyYAML
```

- [ ] **Step 6: Run metadata test and verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest tests/test_dependency_metadata.py -q`

Expected: PASS.

- [ ] **Step 7: Commit Task 5**

Run:

```bash
git add pyproject.toml setup.py requirements.txt tests/test_dependency_metadata.py
git commit -m "fix: align dependency metadata"
```

Expected: commit succeeds if the user wants commits during execution. If the working tree has unrelated user changes in these files, skip the commit and report the conflict.

---

### Task 6: Final Verification

**Files:**
- No production edits unless a verification failure identifies a root cause in earlier tasks.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest \
  tests/experiments/test_registry.py \
  tests/cli/test_exp_cli.py \
  tests/experiments/test_optional_wandb.py \
  tests/experiments/test_imputation_runtime.py \
  tests/test_dependency_metadata.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run collection check**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest --collect-only tests -q`

Expected: new lightweight tests are collected. Existing dataset-heavy tests may also be collected if their filenames match pytest defaults.

- [ ] **Step 3: Run import smoke test**

Run: `PYTHONDONTWRITEBYTECODE=1 python -c "from torch_timeseries.experiments.registry import get_experiment_class; print(get_experiment_class('DLinear', 'Forecast').__name__)"`

Expected output includes:

```text
DLinearForecast
```

- [ ] **Step 4: Report residual risks**

Include these notes in the final implementation summary:

```text
Residual risk: Existing integration tests still instantiate real datasets and may download data.
Residual risk: Full training paths were not executed in focused unit tests.
Residual risk: Dependency install verification requires an environment with the declared packages installed.
```

---

## Self-Review

- Spec coverage: The plan covers the prior review findings around CLI dispatch, optional wandb failure paths, imputation runtime logging, metadata drift, and the lack of lightweight tests.
- Placeholder scan: No task contains `TBD`, `TODO`, or "write tests" without exact test code.
- Type consistency: Registry helper names are `EXPERIMENT_REGISTRY`, `available_experiments`, and `get_experiment_class` throughout. wandb helper name is `_require_wandb` in each touched experiment module.
