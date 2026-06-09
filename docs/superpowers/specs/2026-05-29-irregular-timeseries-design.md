# Irregular Time Series Benchmarks Design Spec

**Date:** 2026-05-29

---

## Goal

Add benchmarks and baselines for irregular time series — sequences with non-uniform observation times and per-feature missingness — supporting three tasks: classification, interpolation, and forecasting. Integrates with the existing `DataModule` + `Experiment` + `register_model` pattern so results flow into `LocalBackend` and `pytexp compare`.

---

## Phased Delivery

| Phase | Scope | Ships standalone |
|-------|-------|-----------------|
| 1 | `IrregularTSBatch` + `collate_irregular` + PhysioNet2012 + P19 + `IrregularClassificationDataModule` + GRU-D + `IrregularClassificationExp` | ✅ |
| 2 | Interpolation + Forecast DataModules + MIMIC + `UEAIrregular` + `IrregularWrapper` | ✅ |
| 3 | mTAN, LatentODE, NeuralCDE, Raindrop (lazy-import models) | ✅ |

Each phase produces working, testable software on its own.

---

## Section 1 — Data Representation

### `IrregularTSBatch`

Replaces `TSBatch` for irregular tasks. Lives in `torch_timeseries/dataloader/v2/irregular_batch.py`.

```python
@dataclass
class IrregularTSBatch:
    x:            Tensor                   # (B, T, F)  padded observed values
    t:            Tensor                   # (B, T)     elapsed time normalized to [0, 1]
    mask:         Tensor                   # (B, T, F)  1=observed, 0=missing or padded
    x_time:       Optional[Tensor] = None  # (B, T, C)  calendar features at obs times
    y:            Optional[Tensor] = None  # (B,) class label OR (B, Tq, F) query targets
    t_query:      Optional[Tensor] = None  # (B, Tq)    query times (interp / forecast)
    query_mask:   Optional[Tensor] = None  # (B, Tq, F) which queries to evaluate
    t_query_time: Optional[Tensor] = None  # (B, Tq, C) calendar features at query times
```

**Time normalization:** each sample's `t` vector is normalized to `[0, 1]` independently — `t_norm = (t - t_min) / (t_max - t_min + ε)`. This makes models scale-invariant across datasets with different time units (hours, minutes, days).

**Padding:** sequences are padded to the longest sequence in the batch. Padded positions have `mask == 0` and `x == 0`. `t` at padded positions is set to `1.0` (end of normalized range, safely outside any real observation).

### `collate_irregular`

Custom collate function used by all three DataModules. Takes a list of `IrregularTSBatch` (one per sample, variable `T`) and returns a single batched `IrregularTSBatch` with padded tensors.

```python
def collate_irregular(samples: List[IrregularTSBatch]) -> IrregularTSBatch:
    ...
```

Lives in `torch_timeseries/dataloader/v2/irregular_batch.py` alongside the dataclass.

---

## Section 2 — Datasets

### Base Class

`IrregularTimeSeriesDataset` lives in `torch_timeseries/dataset/irregular/base.py`.

```python
class IrregularTimeSeriesDataset:
    samples:      List[np.ndarray]          # [(T_i, F)] variable-length per sample
    times:        List[np.ndarray]          # [(T_i,)] raw observation times
    masks:        List[np.ndarray]          # [(T_i, F)] 1=observed
    labels:       Optional[np.ndarray]      # (N,) integer class labels
    num_features: int
    num_classes:  int

    def download(self) -> None: ...
    def _load(self) -> None: ...
```

Subclasses implement `download()` and `_load()`. `root` is passed to `__init__` and is where data is cached after download.

### Concrete Datasets

**`PhysioNet2012`** (`torch_timeseries/dataset/irregular/physionet2012.py`)
- Source: PhysioNet Challenge 2012 public set-a/b/c
- 12,000 ICU patient records, 41 variables, up to 48 hours
- Label: in-hospital mortality (binary)
- Auto-download via `urllib` from `https://physionet.org/files/challenge-2012/1.0.0/`
- `num_features=41`, `num_classes=2`

**`PhysioNet2019`** (`torch_timeseries/dataset/irregular/physionet2019.py`)
- Source: PhysioNet Challenge 2019 (sepsis prediction)
- ~40,000 patients, 40 variables
- Label: sepsis onset (binary)
- Auto-download from PhysioNet public challenge page
- `num_features=40`, `num_classes=2`

**`MIMIC`** (`torch_timeseries/dataset/irregular/mimic.py`)
- Cannot auto-download (requires credentialed PhysioNet access)
- Constructor: `MIMIC(data_dir="/path/to/mimic")` — raises `FileNotFoundError` with instructions if directory missing
- Supports MIMIC-III and MIMIC-IV via a `version` parameter

**`UEAIrregular`** (`torch_timeseries/dataset/irregular/uea_irregular.py`)
- Wraps any existing UEA dataset and applies synthetic timestamp dropout
- Constructor: `UEAIrregular(name="EthanolConcentration", root="./data", drop_rate=0.3, seed=42)`
- Converts regular UEA samples to irregular by randomly removing observations

**`IrregularWrapper`** (`torch_timeseries/dataset/irregular/wrapper.py`)
- Wraps any `TimeSeriesDataset` (ETTh1, Weather, etc.) and applies random timestamp dropout to sliding windows
- Constructor: `IrregularWrapper(dataset=ETTh1("./data"), drop_rate=0.5, seed=42)`
- Used primarily with `IrregularForecastDataModule`

### File Map

```
torch_timeseries/dataset/irregular/
  __init__.py
  base.py
  physionet2012.py
  physionet2019.py
  mimic.py
  uea_irregular.py
  wrapper.py
```

---

## Section 3 — DataModules

All three DataModules live in `torch_timeseries/dataloader/v2/`. All follow `_fit_scaler` / `_build_datasets` / `_build_loaders` pattern. All use `collate_irregular` as the DataLoader collate function.

### Task-specific configs

```python
@dataclass
class IrregularClassificationConfig:
    time_enc: int = 0           # 0=none, 1=fourier (TimeEncoding)
    freq: Optional[str] = None  # required if time_enc > 0

@dataclass
class IrregularInterpolationConfig:
    query_rate: float = 0.2     # fraction of observations held out as query targets
    time_enc: int = 0
    freq: Optional[str] = None

@dataclass
class IrregularForecastConfig:
    obs_frac: float = 0.7       # fraction of each sample's timespan used as input
    time_enc: int = 1
    freq: Optional[str] = None
```

### `IrregularClassificationDataModule`

```python
IrregularClassificationDataModule(
    dataset=PhysioNet2012("./data"),
    scaler=StandardScaler(),
    window=IrregularClassificationConfig(time_enc=0),
    split=SplitConfig(train=0.7, val=0.1, test=0.2),
    loader=LoaderConfig(batch_size=32),
)
# train/val/test: IrregularTSBatch(x, t, mask, x_time=None, y=class_label)
```

Scaler is fitted on the concatenated observed values of training samples only. Val/test samples are transformed but never fitted.

Properties: `num_features`, `num_classes`, `train_loader`, `val_loader`, `test_loader`, `train_dataset`, `val_dataset`, `test_dataset`.

### `IrregularInterpolationDataModule`

```python
IrregularInterpolationDataModule(
    dataset=PhysioNet2012("./data"),
    scaler=StandardScaler(),
    window=IrregularInterpolationConfig(query_rate=0.2),
    split=SplitConfig(train=0.7, val=0.1, test=0.2),
    loader=LoaderConfig(batch_size=32),
)
# IrregularTSBatch(x, t, mask, y=query_values, t_query, query_mask)
```

For each sample, `query_rate` fraction of observed time points are randomly held out. The remaining points form `(x, t, mask)`; the held-out points form `(t_query, y, query_mask)`. Query selection is deterministic per sample (seeded by sample index) so val/test are reproducible.

Properties: `num_features`, `query_rate`, `train_loader`, `val_loader`, `test_loader`.

### `IrregularForecastDataModule`

```python
IrregularForecastDataModule(
    dataset=IrregularWrapper(ETTh1("./data"), drop_rate=0.5),
    scaler=StandardScaler(),
    window=IrregularForecastConfig(obs_frac=0.7, time_enc=1, freq="h"),
    split=SplitConfig(train=0.7, val=0.1, test=0.2),
    loader=LoaderConfig(batch_size=32),
)
# IrregularTSBatch(x, t, mask, x_time, y=future_values, t_query, t_query_time, query_mask)
```

Each sample is split at `obs_frac` of its total timespan: observations before the split point form the input; observations after form the forecast targets. `query_mask` is all-ones (all future points are targets).

Properties: `num_features`, `obs_frac`, `train_loader`, `val_loader`, `test_loader`.

### File Map

```
torch_timeseries/dataloader/v2/
  irregular_batch.py          # IrregularTSBatch, collate_irregular
  irregular_classification.py # IrregularClassificationDataModule + Config
  irregular_interpolation.py  # IrregularInterpolationDataModule + Config
  irregular_forecast.py       # IrregularForecastDataModule + Config
```

Exports added to `torch_timeseries/dataloader/v2/__init__.py`.

---

## Section 4 — Models

All models live in `torch_timeseries/model/irregular/`. Each is a standalone `nn.Module`.

### Model interfaces

All classification models share:
```python
class IrregularClassifier(nn.Module):
    def forward(self, x, t, mask, x_time=None) -> Tensor:
        # returns (B, num_classes) logits
```

All interpolation/forecasting models share:
```python
class IrregularSeq2Seq(nn.Module):
    def forward(self, x, t, mask, t_query, x_time=None, t_query_time=None) -> Tensor:
        # returns (B, Tq, F) predictions at query times
```

### GRU-D (`grud.py`) — no external deps

GRU with exponential decay on both the input and hidden state when observations are missing. Tracks last-observation time per feature. Outputs final hidden state for classification or per-step hidden states projected to feature space.

Key parameters: `input_size`, `hidden_size`, `output_size`, `dropout`.

### mTAN (`mtan.py`) — no external deps

Multi-Time Attention Network. Learns a set of reference time points and uses learned time embeddings to compute attention weights between query times and reference times. Encoder maps irregular observations to a fixed-size latent; decoder queries at arbitrary times.

Key parameters: `input_size`, `hidden_size`, `num_ref_points`, `num_heads`.

### LatentODE (`latent_ode.py`) — requires `torchdiffeq`

Variational latent ODE: RNN encoder over reversed time sequence → latent `z0` → ODE solver forward in time → decoder at query times.

Lazy import:
```python
try:
    from torchdiffeq import odeint
except ImportError:
    raise ImportError("LatentODE requires torchdiffeq: pip install torch-timeseries[irregular]")
```

Key parameters: `input_size`, `latent_size`, `hidden_size`, `ode_method` (default `"dopri5"`).

### NeuralCDE (`neural_cde.py`) — requires `torchcde`

Controlled differential equation driven by a natural cubic spline fitted to the irregular observations. Used primarily for classification (fixed-length output from terminal hidden state).

Lazy import from `torchcde`.

Key parameters: `input_size`, `hidden_size`, `output_size`, `interpolation` (default `"cubic"`).

### Raindrop (`raindrop.py`) — requires `torch_geometric`

Graph-based sensor network where each feature is a node; attention between sensor nodes depends on temporal proximity and feature correlations.

Lazy import from `torch_geometric`.

Key parameters: `input_size`, `hidden_size`, `num_nodes`, `num_heads`, `dropout`.

### Optional extras in `pyproject.toml`

```toml
[project.optional-dependencies]
irregular = [
    "torchdiffeq>=0.2.3",
    "torchcde>=0.2.5",
    "torch_geometric>=2.0.0",
]
```

### File Map

```
torch_timeseries/model/irregular/
  __init__.py
  grud.py
  mtan.py
  latent_ode.py
  neural_cde.py
  raindrop.py
```

---

## Section 5 — Experiment Wiring

### Task mixins

Three new experiment base classes in `torch_timeseries/experiments/`:

**`IrregularClassificationExp`** (`irregular_classification.py`)
- Loss: `CrossEntropyLoss`
- `_prepare_batch(batch: IrregularTSBatch)` → `(x, t, mask, x_time, y)`
- Metrics: accuracy, AUROC (via `torchmetrics`)
- Optimizer: RAdam (matches `UEAClassificationExp`)
- Scheduler: `CosineAnnealingLR(T_max=epochs)`

**`IrregularInterpolationExp`** (`irregular_interpolation.py`)
- Loss: `MSELoss` on `query_mask == 1` positions only
- `_prepare_batch` → `(x, t, mask, t_query, y, query_mask)`
- Metrics: MSE, MAE on query points
- Optimizer: Adam, `CosineAnnealingLR`

**`IrregularForecastExp`** (`irregular_forecast.py`)
- Loss: `MSELoss` on all future query points
- `_prepare_batch` → `(x, t, mask, x_time, t_query, t_query_time, y, query_mask)`
- Metrics: MSE, MAE
- Optimizer: Adam, `CosineAnnealingLR`

### Combo classes + registry

Model-specific files auto-generate combo classes by combining model parameter dataclass with task exp:

```
torch_timeseries/experiments/GRUD.py
  GRUDIrregularClassification(IrregularClassificationExp, GRUDParameters)
  GRUDIrregularInterpolation(IrregularInterpolationExp, GRUDParameters)
  GRUDIrregularForecast(IrregularForecastExp, GRUDParameters)
```

Same pattern for mTAN, LatentODE, NeuralCDE, Raindrop.

All combo classes are imported in `torch_timeseries/experiments/__init__.py` so `build_experiment_registry(globals())` picks them up automatically.

### Experiment builder — unchanged

```python
Experiment(model="GRUD", task="IrregularClassification", dataset="PhysioNet2012").run(seeds=[1,2,3])
Experiment.grid(
    models=["GRUD", "mTAN"],
    tasks=["IrregularClassification"],
    datasets=["PhysioNet2012", "PhysioNet2019"],
    seeds=[1, 2, 3],
).run()
Experiment.compare(save_dir="./results", task="IrregularClassification")
```

---

## Section 6 — Testing Strategy

| Test | File | Verifies |
|------|------|---------|
| `test_irregular_tsbatch_collation` | `tests/dataloader/test_v2_irregular.py` | variable-length batch pads correctly, mask shape, t normalized to [0,1] |
| `test_physionet2012_loads` | `tests/dataset/test_irregular_datasets.py` | dataset loads, `samples`/`times`/`masks`/`labels` shapes |
| `test_irregular_classification_dm_returns_batch` | `tests/dataloader/test_v2_irregular.py` | `IrregularClassificationDataModule` returns `IrregularTSBatch` with correct field shapes |
| `test_irregular_interpolation_dm_query_holdout` | same | held-out query points not in input mask |
| `test_irregular_forecast_dm_future_split` | same | all `t_query` values > max input `t` |
| `test_grud_forward_classification` | `tests/model/test_irregular_models.py` | GRU-D returns `(B, num_classes)` on synthetic batch |
| `test_grud_forward_interpolation` | same | GRU-D returns `(B, Tq, F)` on synthetic batch |
| `test_latent_ode_lazy_import_error` | same | importing LatentODE without torchdiffeq raises `ImportError` with install hint |
| `test_irregular_classification_exp_single_run` | `tests/experiments/test_irregular_experiments.py` | `Experiment(model="GRUD", task="IrregularClassification", dataset="PhysioNet2012").run(seeds=[1])` returns `RunResult` |

Tests for Phase 1 only use synthetic data (no network required). A `_ToyIrregular` fixture generates random variable-length sequences in-memory, mirroring the `_ToyTS` pattern from existing tests.

---

## Section 7 — Non-Goals

- No leaderboard UI or remote upload (covered by future `HttpBackend`)
- No probabilistic forecasting or generative models (separate specs)
- No multi-task learning across regular + irregular jointly
- MIMIC auto-download is out of scope (credentialed access)
- `NeuralCDE` and `Raindrop` classification only in Phase 3 (their architectures do not naturally generalize to interpolation/forecasting without significant redesign)
