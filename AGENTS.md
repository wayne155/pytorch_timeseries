# AGENTS.md — torch-timeseries

All-in-one deep learning library for time series research. Current version: **0.2.3**.

---

## Project layout

```
torch_timeseries/         # main package
  dataset/                # built-in datasets (ETTh1/h2/m1/m2, Weather, Electricity,
  │                       #   M4, UEA, PhysioNet 2012/2019, custom CSV, …)
  │   irregular/          # irregular time-series datasets (PhysioNet*)
  dataloader/
  │   v2/                 # modern API — ForecastDataModule, IrregularClassificationDataModule,
  │   │                   #   TSBatch, WindowConfig, SplitConfig, LoaderConfig, …
  │   *.py                # legacy dataloaders (SlidingWindowTS, …) — kept for compat
  model/                  # model implementations (DLinear, iTransformer, PatchTST,
  │                       #   Autoformer, FEDformer, Informer, Crossformer, TimesNet,
  │   irregular/          #   SCINet, CATS, FITS, FreTS, NLinear, TSMixer)
  │                       #   + irregular/ (GRU-D)
  experiments/            # wires model + dataloader + training loop
  │   forecast.py         # ForecastExp (standard)
  │   prob_forecast.py    # ProbForecastExp (probabilistic / diffusion)
  │   irregular_classification.py
  │   imputation.py / anomaly_detection.py / uea_classification.py
  │   {Model}.py          # per-model experiment subclasses (DLinear.py, GRUD.py, …)
  │   engine.py           # ForecastEngine (DLinear, Crossformer typed-config path)
  │   registry.py         # auto-discovery: {Model}{Task} naming convention
  │   configs.py          # DLinearConfig, CrossformerConfig, RuntimeConfig
  nn/                     # reusable building blocks (attention, embeddings, decomp, …)
  metrics/                # probabilistic metrics: CRPS, CRPSSum, QICE, PICP, ProbMSE/MAE/RMSE
  results/                # RunResult schema; LocalBackend, WandbBackend, ArtifactBackend
  scaler/                 # StandardScaler, MaxAbsScaler
  leaderboard/            # leaderboard rendering / CLI
  cli/exp.py              # pytexp CLI entry-point
  experiment.py           # Experiment fluent builder + register_model()

leaderboard/reproduce/    # per-task/model scripts to reproduce benchmark numbers
tests/                    # pytest suite
examples/                 # standalone usage examples
```

---

## Key abstractions

### Tasks
Five task types, identified by class-name suffix and auto-discovered by `registry.py`:

| Suffix | Description |
|---|---|
| `Forecast` | Standard multivariate/univariate forecasting |
| `Imputation` | Missing-value imputation |
| `UEAClassification` | UEA archive classification |
| `AnomalyDetection` | Anomaly detection |
| `IrregularClassification` | Irregular / sporadically sampled time series classification |

### Experiment naming convention
A class named `DLinearForecast` is automatically registered as model `DLinear`, task `Forecast`. The registry scans any class that ends in one of the five suffixes.

### v2 dataloader — `TSBatch`
The structured batch type. **Never unpack positionally.** Use named fields:

```python
batch.x            # (B, window, C) — scaled input
batch.y            # (B, pred_len, C) — scaled target
batch.x_raw        # unscaled input
batch.x_time_feature  # (B, window, enc_dim) float32
batch.x_time       # Time namedtuple with .year/.month/… LongTensors
batch.x_index      # (B,) integer position in the dataset
```

### `Experiment` fluent builder
High-level API for running and comparing experiments:

```python
from torch_timeseries import Experiment

Experiment("DLinear", "Forecast", "ETTh1") \
    .set(pred_len=96, epochs=10) \
    .with_local("./results") \
    .run(seed=42)
```

### `register_model` decorator
For adding a custom model into the CLI/experiment registry:

```python
from torch_timeseries import register_model

@register_model
class MyModelForecast(ForecastExp):
    ...
```

---

## Development setup

```bash
pip install -e ".[dev]"   # installs test + pre-commit deps
pre-commit install
```

Core runtime deps: `torch`, `numpy`, `pandas`, `einops`, `sktime>=0.29.0`,
`torchmetrics>=1.1.1`, `fire>=0.5.0`, `PyYAML`, `scikit-learn`.

Optional: `torch_scatter` (`pip install -e ".[full]"`) — needed for graph-based models.

---

## Running tests

```bash
pytest                        # full suite
pytest tests/dataloader/      # dataloader only
pytest tests/experiments/     # experiment integration tests
pytest -k "irregular"         # irregular TS tests
```

Tests are self-contained: toy datasets (`_ToyIrregular`, dummy datasets) are injected
directly so no file I/O or network calls are needed for most tests.

---

## Adding a new model

1. Implement the model in `torch_timeseries/model/{ModelName}.py`.
2. Export it from `torch_timeseries/model/__init__.py`.
3. Create an experiment class in `torch_timeseries/experiments/{ModelName}.py` that
   subclasses the relevant base (`ForecastExp`, `IrregularClassificationExp`, etc.) and
   is named `{ModelName}{Task}`.
4. Import the experiment class in `torch_timeseries/experiments/__init__.py` so the
   registry can discover it.
5. Add leaderboard reproduce scripts under `leaderboard/reproduce/{task}/{modelname}.py`
   following existing examples.
6. Write tests in `tests/experiments/test_{modelname}.py`.

---

## Adding a new dataset

1. Create `torch_timeseries/dataset/{DatasetName}.py` subclassing `TimeSeriesDataset`.
2. Implement `download()` and expose `num_features`, `length`, `freq`, and a default
   split via `DEFAULT_SPLIT_CONFIGS` if applicable.
3. Export from `torch_timeseries/dataset/__init__.py`.
4. (Optional) Register a default split in `torch_timeseries/dataloader/v2/split.py`.

Irregular datasets live in `torch_timeseries/dataset/irregular/` and subclass the base
in `irregular/base.py`.

---

## Results & backends

Results are `RunResult` dataclass instances stored via backends:

| Backend | Usage |
|---|---|
| `LocalBackend(save_dir=...)` | JSON files under `save_dir/` |
| `WandbBackend(project=..., entity=...)` | Weights & Biases |
| `LocalArtifactBackend(save_dir=...)` | model checkpoints |

Run identity is derived from a config fingerprint (MD5 of hyperparams + dataset + model).
Re-running with the same config loads a cached result.

---

## CLI

```bash
pytexp DLinear Forecast ETTh1 --pred_len=96 --epochs=10
```

The CLI is registered as the `pytexp` console script and delegates to
`torch_timeseries/cli/exp.py`. Model/task are resolved through the experiment registry.

---

## Conventions

- No positional tuple unpacking from batches — always use `TSBatch` named fields.
- Scaler always fits on train split only (`scale_in_train` was removed in 0.2.1).
- Default training protocol mirrors Time-Series-Library: 10 epochs, patience 3,
  `lradj=type1` (lr halved each epoch), no gradient clipping.
- Default data dir: `~/.torchtimeseries/data` (override with `root=` or `data_path=`).
- `time_enc` accepts string aliases (`"calendar"`, `"fourier"`, `"normalized"`) or the
  legacy integer 0/1/2; default is `"calendar"` (changed in 0.2.0).
- ETT datasets use canonical calendar splits (12/4/4 months); others default to 7:1:2.
