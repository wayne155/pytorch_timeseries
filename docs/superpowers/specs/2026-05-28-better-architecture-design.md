# Better Architecture for Task Types — Design Spec

**Date:** 2026-05-28

---

## Goal

Replace the current pattern — one experiment class per (model × task) combination (e.g., `AutoformerForecast`, `InformerImputation`) — with a composable mixin architecture: one `BaseExp` class, one task mixin per task type, one model class per model, and auto-generated combo classes. This eliminates ~80% duplicated code, makes adding new models or tasks a one-file change, and produces a tidy CLI/registry surface.

---

## Section 1 — File Structure

### New files

| Path | Responsibility |
|------|---------------|
| `torch_timeseries/experiments/base.py` | `BaseExp` dataclass — shared settings, training loop, logging |
| `torch_timeseries/experiments/mixins/forecast.py` | `ForecastMixin` — data loading, loss, metrics, `_prepare_batch` |
| `torch_timeseries/experiments/mixins/imputation.py` | `ImputationMixin` |
| `torch_timeseries/experiments/mixins/anomaly.py` | `AnomalyMixin` |
| `torch_timeseries/experiments/mixins/uea.py` | `UEAMixin` |
| `torch_timeseries/experiments/mixins/__init__.py` | Re-exports all four mixins |
| `torch_timeseries/experiments/models/dlinear.py` | `DLinear` model class (settings + `_build_model`) |
| `torch_timeseries/experiments/models/autoformer.py` | `Autoformer` |
| `torch_timeseries/experiments/models/informer.py` | `Informer` |
| `torch_timeseries/experiments/models/fedformer.py` | `FEDformer` |
| `torch_timeseries/experiments/models/crossformer.py` | `Crossformer` |
| `torch_timeseries/experiments/models/scinet.py` | `SCINet` |
| `torch_timeseries/experiments/models/itransformer.py` | `iTransformer` |
| `torch_timeseries/experiments/models/tsmixer.py` | `TSMixer` |
| `torch_timeseries/experiments/models/__init__.py` | Re-exports all model classes |

### Modified files

| Path | Change |
|------|--------|
| `torch_timeseries/experiments/__init__.py` | Replace individual imports with combo-generation loop |
| `torch_timeseries/experiments/forecast.py` | Thin shim → delegates to `ForecastMixin`; kept for backward compat |
| `torch_timeseries/experiments/imputation.py` | Same shim pattern |
| `torch_timeseries/experiments/anomaly_detection.py` | Same shim pattern |
| `torch_timeseries/experiments/uea_classification.py` | Same shim pattern |

---

## Section 2 — BaseExp

`BaseExp` is a `@dataclass` that owns everything shared across all tasks and models:

```python
@dataclass
class BaseExp:
    # --- data ---
    dataset_type: str = "ETTh1"
    root_path: str = "./data"
    scaler_type: str = "StandardScaler"
    train_ratio: float = 0.7
    test_ratio: float = 0.2

    # --- training ---
    lr: float = 0.001
    batch_size: int = 32
    num_worker: int = 0
    max_epoch: int = 100
    patience: int = 3
    itr: int = 1

    # --- runtime state (not settings) ---
    model: nn.Module = field(default=None, init=False)
    dataset: TimeSeriesDataset = field(default=None, init=False)
    train_loader: DataLoader = field(default=None, init=False)
    val_loader: DataLoader = field(default=None, init=False)
    test_loader: DataLoader = field(default=None, init=False)
```

**Training loop** lives entirely in `BaseExp._train_one_epoch` / `run()`:

```
run()
  _init_dataset()          ← BaseExp reads dataset_type
  _init_data_loader()      ← Task mixin builds loaders + metrics
  _build_model()           ← Model class defines architecture
  for epoch in range(max_epoch):
      _train_one_epoch()   ← iterates loader, calls _prepare_batch()
      _val()               ← Task mixin defines validation
  _test()                  ← Task mixin defines test
```

`BaseExp` calls abstract hooks; mixins and model classes implement them. No task-specific code lives in `BaseExp`.

---

## Section 3 — Task Mixin Interface

Each task mixin implements exactly this interface:

```python
class ForecastMixin:
    task_suffix = "Forecast"      # used for auto-naming: DLinearForecast

    # Required fields (declared as dataclass fields in the mixin)
    windows: int = 96
    horizon: int = 96
    pred_len: int = 96
    time_enc: int = 1
    input_columns: List[int] = field(default_factory=list)
    target_columns: List[int] = field(default_factory=list)

    def _init_data_loader(self) -> None:
        """Build train/val/test loaders and attach self.train_loader etc."""

    def _prepare_batch(self, batch: TSBatch) -> Tuple[Tensor, ...]:
        """Unpack a TSBatch into model inputs (variable number of tensors)."""

    def _compute_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Return scalar loss."""

    def _init_metrics(self) -> None:
        """Populate self.metrics dict (torchmetrics or plain callables)."""

    def _val(self) -> Dict[str, float]: ...
    def _test(self) -> Dict[str, float]: ...
```

`ImputationMixin`, `AnomalyMixin`, and `UEAMixin` follow the same interface with task-appropriate defaults and implementations.

**Data flow through `_prepare_batch`:**

```
TSBatch → _prepare_batch → (x_enc, x_mark, x_dec, x_dec_mark)  # for transformer models
                         → (x,)                                  # for DLinear / MLP models
```

The mixin knows the data shape; the model class knows the forward signature. The base training loop calls `model(*prepare_batch(batch))`.

---

## Section 4 — Model Classes and Combo Generation

### Model class structure

Each model class is a `@dataclass` that declares only its own hyperparameters and `_build_model`:

```python
# experiments/models/dlinear.py
@dataclass
class DLinear(BaseExp):
    moving_avg: int = 25
    individual: bool = False

    def _build_model(self) -> nn.Module:
        return DLinearModel(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.num_input_features,
            individual=self.individual,
            moving_avg=self.moving_avg,
        )
```

`num_input_features` is a property declared on the task mixin (it knows the column selection from `input_columns` and the dataset). Model classes never import task mixins directly — they reference `self.num_input_features` which is resolved at runtime via MRO.

### Auto-generated combo classes

```python
# experiments/__init__.py
from .models import DLinear, Autoformer, Informer, ...
from .mixins import ForecastMixin, ImputationMixin, AnomalyMixin, UEAMixin

_MODELS = [DLinear, Autoformer, Informer, FEDformer, Crossformer, SCINet, iTransformer, TSMixer]
_TASKS  = [ForecastMixin, ImputationMixin, AnomalyMixin, UEAMixin]

for _m in _MODELS:
    for _t in _TASKS:
        _name = f"{_m.__name__}{_t.task_suffix}"
        globals()[_name] = dataclass(type(_name, (_m, _t), {}))
        __all__.append(_name)
```

MRO: `DLinearForecast → DLinear → ForecastMixin → BaseExp`. Settings from all three are merged by Python's `@dataclass` inheritance.

**Explicit fallback:** For any combo that needs custom behaviour (e.g., `CrossformerForecast` with a non-standard decoder), an explicit class can be defined and placed in `globals()` *before* the loop runs — it simply shadows the auto-generated one.

---

## Section 5 — Backward Compatibility

The existing files (`forecast.py`, `imputation.py`, `anomaly_detection.py`, `uea_classification.py`) become thin shims:

```python
# experiments/forecast.py — backward compat shim
from .base import BaseExp
from .mixins.forecast import ForecastMixin
from dataclasses import dataclass

@dataclass
class ForecastExp(ForecastMixin, BaseExp):
    """Backward-compatible alias. Use model-specific classes for new code."""
    model_type: str = "DLinear"
    ...
```

`ForecastExp` continues to work as before; it just delegates to the new structure. No existing experiment scripts break.

---

## Section 6 — Testing Strategy

| Test | Location | Verifies |
|------|----------|---------|
| `test_base_exp_training_loop` | `tests/experiments/test_base.py` | `run()` calls hooks in correct order |
| `test_forecast_mixin_prepare_batch` | `tests/experiments/test_mixins.py` | `_prepare_batch` output shape |
| `test_combo_class_exists` | `tests/experiments/test_registry.py` | `DLinearForecast` exists in `__init__` |
| `test_combo_class_settings_merge` | same file | settings from both model + mixin present |
| `test_backward_compat_forecast_exp` | `tests/experiments/test_compat.py` | `ForecastExp` still instantiates and runs |
| `test_explicit_override_takes_precedence` | same file | explicit class shadows auto-generated |

---

## Constraints and Non-Goals

- **No breaking changes** to `ForecastExp`, `ImputationExp`, or their existing tests.
- **No change** to the v2 dataloader API (`ForecastDataModule`, `TSBatch`, etc.) — mixin calls it as-is.
- **No new CLI flags** — existing `model_type` / `dataset_type` strings keep working via the registry.
- Out of scope: adding new models, new datasets, or changing training hyperparameters.
