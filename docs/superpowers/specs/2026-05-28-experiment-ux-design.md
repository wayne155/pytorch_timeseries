# Experiment UX Design Spec

**Date:** 2026-05-28

---

## Goal

Make it easy for researchers designing new SOTA models to: (A) define a model with minimal boilerplate and run end-to-end experiments in one command, and (C) compare their model against baselines across datasets and tasks in one command. Results are stored in a leaderboard-ready schema, with local storage as default and W&B as an optional backend.

---

## Section 1 — Overall Architecture

Four layers, each with a single responsibility:

```
┌─────────────────────────────────────────────┐
│  Entry Layer                                 │
│  Experiment(model, task, dataset)            │  ← researcher's only entry point
│  pytexp CLI (thin wrapper)                   │
├─────────────────────────────────────────────┤
│  Experiment Engine                           │
│  BaseExp + Task Mixin + Model Class          │  ← existing architecture design
│  unified training loop, _prepare_batch/_val  │
├─────────────────────────────────────────────┤
│  DataModule Layer                            │
│  ForecastDataModule   (existing)             │
│  ImputationDataModule (new)                  │  ← all return TSBatch
│  AnomalyDataModule    (new)                  │
│  UEADataModule        (new)                  │
├─────────────────────────────────────────────┤
│  Results Layer                               │
│  RunResult (standard JSON schema)            │
│  LocalBackend  → ./results/*.json            │  ← pluggable
│  WandbBackend  → W&B run                     │
│  Experiment.compare() → comparison table     │
└─────────────────────────────────────────────┘
```

**Key decisions:**
- `Experiment` is a builder — it assembles the right `BaseExp` subclass and calls `run()`. It does not contain training logic.
- The DataModule layer is independent of the Experiment Engine and can be used standalone.
- The Results Layer is pure I/O — after training finishes, backends receive a `RunResult` object. Training logic has no knowledge of backends.

---

## Section 2 — DataModule Unification

All four DataModules expose identical properties and return `TSBatch`. Task-specific config classes carry only relevant fields.

**Unified interface (all four DataModules):**

```python
dm = ImputationDataModule(
    dataset=ETTh1("./data"),
    scaler=StandardScaler(),
    window=ImputationWindowConfig(window=96, mask_ratio=0.25, time_enc=1, freq="h"),
    split=SplitConfig(train=0.7, val=0.1, test=0.2),
    loader=LoaderConfig(batch_size=32),
)
dm.train_loader   # DataLoader → TSBatch
dm.val_loader
dm.test_loader
dm.train_dataset
```

**TSBatch semantics per task:**

| Task | `x` | `y` | Notes |
|------|-----|-----|-------|
| Forecast | `(B, window, F)` input | `(B, pred_len, F)` target | existing |
| Imputation | `(B, window, F)` with mask applied | `(B, window, F)` full original | mask generated in `_prepare_batch` |
| AnomalyDetection | `(B, window, F)` input | `None` (train) / label tensor (test) | unsupervised training |
| UEAClassification | `(B, T, F)` time series | `(B,)` class labels | |

**Task-specific window configs (only relevant fields):**

```python
@dataclass
class ImputationWindowConfig:
    window: int = 96
    mask_ratio: float = 0.25
    time_enc: int = 1
    freq: str = "h"

@dataclass
class AnomalyWindowConfig:
    window: int = 96
    stride: int = 1
    time_enc: int = 1
    freq: str = "h"

@dataclass
class UEAWindowConfig:
    normalize: bool = True
```

`SplitConfig` and `LoaderConfig` are reused across all four tasks unchanged.

**New files:**

| Path | Responsibility |
|------|---------------|
| `torch_timeseries/dataloader/v2/imputation.py` | `ImputationDataModule` |
| `torch_timeseries/dataloader/v2/anomaly.py` | `AnomalyDataModule` |
| `torch_timeseries/dataloader/v2/uea.py` | `UEADataModule` |

---

## Section 3 — Experiment Builder API

**Single-run:**

```python
from torch_timeseries import Experiment

result = (
    Experiment(model="DLinear", task="Forecast", dataset="ETTh1")
    .set(windows=96, pred_len=96, lr=0.001)   # override any exp parameter
    .with_local(save_dir="./results")           # optional
    .with_wandb(project="my-sota")              # optional
    .run(seeds=[1, 2, 3])                       # returns RunResult
)
print(result.summary())   # MSE/MAE mean ± std
```

**Grid / batch comparison:**

```python
Experiment.grid(
    models=["DLinear", "Autoformer", "MyNewModel"],
    tasks=["Forecast"],
    datasets=["ETTh1", "ETTm1", "Weather"],
    seeds=[1, 2, 3],
    save_dir="./results",
).run()

# reads saved results, prints comparison table
Experiment.compare(save_dir="./results", task="Forecast")
```

**CLI (unchanged surface, delegates to Experiment internally):**

```bash
pytexp --model DLinear --task Forecast --dataset_type ETTh1 --pred_len 96 run 3
pytexp compare --save_dir ./results --task Forecast
```

**Registering a custom model (minimum interface for researchers):**

```python
# my_model.py
from dataclasses import dataclass
from torch_timeseries.experiments import BaseExp
from torch import nn

@dataclass
class MyModel(BaseExp):
    hidden_dim: int = 64

    def _build_model(self) -> nn.Module:
        return MyModelNet(
            seq_len=self.windows,
            pred_len=self.pred_len,
            hidden_dim=self.hidden_dim,
        )

from torch_timeseries import register_model
register_model(MyModel)

# immediately runnable
Experiment(model="MyModel", task="Forecast", dataset="ETTh1").run(seeds=[1])
```

`register_model` is a plain function (not a decorator) that inserts the class into the global registry. `Experiment` looks up models by string name from that registry.

**New files:**

| Path | Responsibility |
|------|---------------|
| `torch_timeseries/experiment.py` | `Experiment` builder class |
| `torch_timeseries/results/schema.py` | `RunResult` dataclass |
| `torch_timeseries/results/backends.py` | `ResultBackend`, `LocalBackend`, `WandbBackend` |
| `torch_timeseries/results/__init__.py` | re-exports |
| `torch_timeseries/__init__.py` | expose `Experiment`, `register_model` at top level |

**Modified files:**

| Path | Change |
|------|--------|
| `torch_timeseries/cli/exp.py` | delegate to `Experiment` builder; add `compare` subcommand |

---

## Section 4 — Results Schema and Backends

**`RunResult` — leaderboard-ready dataclass:**

```python
@dataclass
class RunResult:
    # experiment identity
    model: str           # "DLinear"
    task: str            # "Forecast"
    dataset: str         # "ETTh1"
    seed: int            # 3
    timestamp: str       # "2026-05-28T14:32:00" (ISO 8601)

    # full hyperparameter snapshot for reproducibility
    hparams: dict        # {"windows": 96, "pred_len": 96, "lr": 0.001, ...}

    # metrics
    metrics: dict        # {"mse": 0.382, "mae": 0.271, "val_mse": 0.401}

    # run metadata
    num_params: int
    train_time_sec: float
    git_commit: str      # auto-captured via `git rev-parse HEAD`

    # reserved for future use (training curves, leaderboard)
    history: Optional[dict] = None   # None in this version
```

**Local storage structure:**

```
results/
  DLinear_Forecast_ETTh1_seed1.json
  DLinear_Forecast_ETTh1_seed2.json
  Autoformer_Forecast_ETTh1_seed1.json
  MyNewModel_Forecast_ETTh1_seed1.json
  ...
```

One file per seed per run. Plain JSON, human-readable, git-friendly. Leaderboard reads this directory directly.

**Backend interface:**

```python
class ResultBackend:
    def save(self, result: RunResult) -> None: ...
    def load_all(self, save_dir: str, **filters) -> List[RunResult]: ...

class LocalBackend(ResultBackend):
    def __init__(self, save_dir: str = "./results"): ...

class WandbBackend(ResultBackend):
    def __init__(self, project: str, entity: str = None): ...
```

**`Experiment.compare()` output:**

```
Task: Forecast | Dataset: ETTh1 | pred_len=96
─────────────────────────────────────────────
Model         MSE (avg±std)    MAE (avg±std)    #params
DLinear       0.382 ±0.003     0.271 ±0.002     22K
Autoformer    0.421 ±0.008     0.301 ±0.005     8.2M
MyNewModel    0.361 ±0.002     0.255 ±0.001     145K
```

**Future extension path (out of scope for this plan, schema already accommodates):**

- `leaderboard.json` = aggregated `RunResult` list, served by a leaderboard page
- Visualization: training curves read from `result.history` once populated
- `HttpBackend(url=...)` for uploading to a public leaderboard

---

## Section 5 — Constraints and Non-Goals

- **No breaking changes** to existing `ForecastExp`, `ImputationExp`, `DLinearForecast`, etc. — they remain usable as-is.
- **No new training logic** — all training logic lives in the Experiment Engine (BaseExp + mixins); the builder just wires it up.
- **Out of scope:** leaderboard UI, visualization, remote backend, `history` field population, new models or datasets.
- The `Experiment` builder resolves `model` / `task` / `dataset` strings to classes via the existing registry (`get_experiment_class`).

---

## Section 6 — Testing Strategy

| Test | File | Verifies |
|------|------|---------|
| `test_imputation_datamodule_returns_tsbatch` | `tests/dataloader/test_v2_task_modules.py` | `ImputationDataModule` batch shape and type |
| `test_anomaly_datamodule_returns_tsbatch` | same | `AnomalyDataModule` |
| `test_uea_datamodule_returns_tsbatch` | same | `UEADataModule` |
| `test_experiment_builder_single_run` | `tests/test_experiment_builder.py` | `Experiment(...).run()` returns `RunResult` with correct fields |
| `test_experiment_grid_runs_all_combos` | same | all model×task×dataset combos executed |
| `test_local_backend_saves_and_loads` | `tests/results/test_backends.py` | `LocalBackend.save` writes JSON; `load_all` reads with filters |
| `test_run_result_schema_fields` | same | all required fields present in saved JSON |
| `test_register_model_makes_it_runnable` | `tests/test_experiment_builder.py` | `register_model` + `Experiment(model="MyModel", ...)` succeeds |
| `test_compare_prints_table` | same | `Experiment.compare()` produces non-empty output |
| `test_cli_compare_subcommand` | `tests/test_cli_and_registry.py` | `pytexp compare` exits 0 |
