[pypi-image]: https://badge.fury.io/py/torch-timeseries.svg
[pypi-url]: https://pypi.python.org/pypi/torch-timeseries
[docs-image]: https://readthedocs.org/projects/pytorch-timeseries/badge/?version=latest
[docs-url]: https://pytorch-timeseries.readthedocs.io/en/latest/?badge=latest



<p align="center">
  <img width="90%" src="https://raw.githubusercontent.com/wayne155/pytorch_timeseries/main/docs/_static/img/logo_text.jpg?sanitize=true" />
</p>

[![PyPI Version][pypi-image]][pypi-url]
[![Docs Status][docs-image]][docs-url]



# pytorch_timeseries

An all-in-one deep learning library covering the full spectrum of time series research tasks — **forecasting, probabilistic forecasting, imputation, anomaly detection, classification, generation, and irregular time series** — with datasets that download automatically, a highly customisable data pipeline, and a one-command experiment runner.
[Full documentation](https://pytorch-timeseries.readthedocs.io/en/latest/).

---

## Table of Contents

- [Installation](#installation)
- [Quick start — your own CSV](#quick-start--your-own-csv)
- [Model Training — Two Ways to Use](#model-training--two-ways-to-use)
  - [Way 1 — Custom pipeline](#way-1--custom-pipeline-bring-your-own-training-loop)
  - [Way 2 — Built-in experiment runner](#way-2--default-training-paradigm-built-in-or-registered-models)
- [Datasets](#datasets)
  - [Custom Datasets](#custom-datasets)
  - [Fast evaluation windows](#fast-evaluation-windows-fast_val--fast_test)
- [Time Series Tasks](#time-series-tasks)
  - [Forecasting](#forecasting)
  - [Probabilistic Forecasting](#probabilistic-forecasting)
  - [Time Series Generation](#time-series-generation)
  - [Imputation · Anomaly Detection · Classification](#imputation--anomaly-detection--classification)
- [Development Milestones](#development-milestones)
  - [Implemented Datasets](#implemented-datasets)
  - [Implemented Tasks](#implemented-tasks)
  - [Implemented Models](#implemented-models)
- [Dev Install](#dev-install)

---

## Installation

```bash
pip install torch-timeseries
```

> **Python 3.8+ required.**



## Quick start — your own CSV

Have a CSV file and want forecasts in minutes? Three steps:

```
my_data.csv
─────────────────────────────
date,        temp,  humidity
2023-01-01,  12.3,  65.1
2023-01-02,  13.1,  67.4
...
```

```python
import torch, torch.nn as nn
from torch_timeseries.dataset import build_dataset
from torch_timeseries.scaler import StandardScaler
from torch_timeseries.dataloader.v2 import ForecastDataModule, WindowConfig
from torch_timeseries.model import DLinear

# 1. Load — needs a 'date' column; every other column becomes a feature
dataset = build_dataset(csv="my_data.csv", freq="h")

# 2. Wrap — 96-step look-back, predict the next 24 steps
dm = ForecastDataModule(
    dataset=dataset,
    scaler=StandardScaler(),
    window=WindowConfig(window=96, horizon=1, steps=24),
)

# 3. Train
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = DLinear(seq_len=96, pred_len=24, enc_in=dataset.num_features).to(device)
opt    = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    for batch in dm.train_loader:
        x, y = batch.x.float().to(device), batch.y.float().to(device)
        opt.zero_grad(); nn.MSELoss()(model(x), y).backward(); opt.step()

# 4. Predict
model.eval()
with torch.no_grad():
    batch = next(iter(dm.test_loader))
    preds = model(batch.x.float().to(device))  # (B, 24, num_features)
    print(preds.shape)  # → torch.Size([32, 24, 2])
```

Swap `DLinear` for any of the [20 built-in models](https://pytorch-timeseries.readthedocs.io/en/latest/modules/model.html) — the interface is the same.

---

## Model Training — Two Ways to Use

### Way 1 — Custom pipeline (bring your own training loop)

Import a dataset and dataloader, then write your own training logic. Full control over loss, optimizer, and batch handling.

```python
import torch
import torch.nn as nn

from torch_timeseries.dataset import ETTh1
from torch_timeseries.scaler import StandardScaler
from torch_timeseries.dataloader.v2 import (
    ForecastDataModule, WindowConfig, SplitConfig, LoaderConfig
)

# Dataset is downloaded automatically on first use
# (to ~/.torchtimeseries/data by default; pass a path to override).
dataset = ETTh1()

dm = ForecastDataModule(
    dataset=dataset,
    scaler=StandardScaler(),
    window=WindowConfig(window=96, horizon=1, steps=96),
    # ETTh1 academic split: 12 months train, 4 months val, 4 months test.
    # If split is omitted, the datamodule uses this dataset default.
    split=SplitConfig(borders=(12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24)),
    loader=LoaderConfig(batch_size=32),
)

class LinearForecaster(nn.Module):
    """Input: (batch, input_window, features). Output: (batch, pred_len, features)."""

    def __init__(self, input_window: int, pred_len: int):
        super().__init__()
        self.proj = nn.Linear(input_window, pred_len)

    def forward(self, x):
        # x: (B, 96, C) -> (B, C, 96) -> (B, C, 96) -> (B, 96, C)
        return self.proj(x.transpose(1, 2)).transpose(1, 2)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LinearForecaster(input_window=96, pred_len=96).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(1):
    model.train()
    for batch in dm.train_loader:
        # Each batch is a TSBatch.
        x = batch.x.float().to(device)  # (B, 96, num_features)
        y = batch.y.float().to(device)  # (B, 96, num_features)

        optimizer.zero_grad()
        pred = model(x)                 # (B, 96, num_features)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
```

Grey = input window · Blue dashed = ground truth · Orange = forecast:

![LinearForecaster predictions](docs/_static/img/forecast_custom_pipeline.png)

Use this pattern when you need a non-standard training loop, custom loss, or are prototyping a new architecture.

---

### Way 2 — Default training paradigm (built-in or registered models)

Use the built-in experiment runner. Pick a model, task, and dataset — the library handles data loading, training, evaluation, and result saving.

This path works for built-in models and for your own models registered with the default experiment classes.

#### Architecture Direction

New development targets the v2 DataModule API and the high-level `Experiment`
entrypoint. Legacy dataloaders and direct experiment classes remain available
for compatibility, but new task/model features should use named batches,
Task DataModules, and result records.



#### Register Custom Models

To use the default training loop with your own model, subclass the task experiment class, define `_init_model`, then register it.

For forecasting, the model should read `batch_x` with shape `(batch, windows, num_features)` and return predictions with shape `(batch, pred_len, num_features)`.

```python
from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_timeseries import Experiment, register_model
from torch_timeseries.experiments import ForecastExp


class MyForecastNet(nn.Module):
    """Input: (B, seq_len, C). Output: (B, pred_len, C)."""

    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.proj = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        return self.proj(x.transpose(1, 2)).transpose(1, 2)


@dataclass
class MyForecastModel(ForecastExp):
    model_type: str = "MyForecastModel"

    def _init_model(self):
        self.model = MyForecastNet(
            seq_len=self.windows,
            pred_len=self.pred_len,
        ).to(self.device)


register_model(MyForecastModel)

# The registered model name is the class name.
device = "cuda" if torch.cuda.is_available() else "cpu"

results = Experiment(
    model="MyForecastModel",
    task="Forecast",
    dataset="ETTh1",
    windows=96,
    pred_len=96,
    epochs=1,
    device=device,
).run(seeds=[1])

print(results[0].metrics)
```

The same registered model can be launched from the CLI after the Python module containing `register_model(...)` has been imported:

```bash
pytexp --model MyForecastModel --task Forecast --dataset_type ETTh1 run 1
```

#### Run Built-In Models

**Experiment builder (Python API):**

```python
from torch_timeseries import Experiment

# single run — returns a RunResult with metrics, hparams, git commit, timing
result = Experiment(model="DLinear", task="Forecast", dataset="ETTh1").run(seeds=[1])
print(result[0].metrics)   # {"mse": 0.382, "mae": 0.271}

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

# grid search across models and datasets
Experiment.grid(
    models=["DLinear", "Autoformer"],
    tasks=["Forecast"],
    datasets=["ETTh1", "ETTm1"],
    seeds=[1, 2, 3],
    save_dir="./results",
).run()

# compare saved results
Experiment.compare(save_dir="./results", task="Forecast")
```

**CLI:**

```bash
# forecast
pytexp --model DLinear --task Forecast --dataset_type ETTh1 run 3
pytexp --model DLinear --task Forecast --dataset_type ETTh1 runs '[1,2,3]'

# imputation
pytexp --model DLinear --task Imputation --dataset_type ETTh1 run 3

# anomaly detection
pytexp --model DLinear --task AnomalyDetection --dataset_type MSL run 3

# classification
pytexp --model DLinear --task UEAClassification --dataset_type EthanolConcentration run 3

# compare saved results
pytexp compare --save_dir ./results --task Forecast
```

Representative ETTh1 forecasting metrics (pred_len=96) across built-in models:

![Experiment builder — ETTh1 benchmark metrics](docs/_static/img/experiment_builder.png)

Use this pattern when you want to benchmark on standard tasks without writing boilerplate.

---

## Datasets

### Custom Datasets

The quickest path is a local CSV with a `date` column — everything else
(feature count, length, time index) is inferred from the file:

```python
from torch_timeseries.dataset import build_dataset

dataset = build_dataset(csv="./my_sensors.csv", freq="h")
dm = ForecastDataModule(dataset=dataset, scaler=StandardScaler(),
                        window=WindowConfig(window=96, steps=96))
```

For datasets that need downloading or preprocessing, subclass
`TimeSeriesDataset` and implement `download()` and `_load()`. The contract is
small: `_load()` must set `self.df` (a DataFrame with a `date` column),
`self.dates`, and `self.data` (numpy array `[T, num_features]`) —
`num_features` and `length` are inferred from the loaded data.

```python
import os
import numpy as np
import pandas as pd

from torch_timeseries.core import TimeSeriesDataset, Freq

class MySensors(TimeSeriesDataset):
    name: str = "MySensors"        # subdirectory under the data root
    freq: Freq = "h"               # used by time-feature encoding
    # Optional canonical benchmark split: register (train_end, val_end,
    # test_end) in torch_timeseries.dataloader.v2.split.DEFAULT_SPLIT_CONFIGS;
    # without it, dataloaders fall back to the 7:1:2 ratio split.

    def download(self):
        # Fetch raw files into self.dir, or no-op if the data is already local.
        pass

    def _load(self) -> np.ndarray:
        self.file_path = os.path.join(self.dir, "my_sensors.csv")
        # CSV layout: a `date` column + one column per variable
        self.df = pd.read_csv(self.file_path, parse_dates=["date"])
        self.dates = pd.DataFrame({"date": self.df.date})
        self.data = self.df.drop("date", axis=1).to_numpy()
        return self.data

# Works everywhere a built-in dataset works:
dataset = MySensors()              # stored at ~/.torchtimeseries/data/MySensors
```

### Fast evaluation windows (`fast_val` / `fast_test`)

Training always slides the window one step at a time, but evaluating every
overlapping window is wasteful when inference is expensive — a diffusion
model sampling 100 trajectories per window would run the sampler thousands
of times. `WindowConfig.fast_val` / `fast_test` switch the val/test split to
**non-overlapping** windows (stride = `window + horizon + steps − 1`) while
training keeps the dense sliding window:

```python
dm = ForecastDataModule(
    dataset=ETTh1(),
    scaler=StandardScaler(),
    window=WindowConfig(window=96, steps=24, fast_val=True, fast_test=True),
)
# ETTh1, pred_len 24:  val/test windows  2857 -> 24  (119x fewer model calls)
```

Blue = input window · Orange = prediction horizon. Top: training (dense). Bottom: eval with `fast_val=True` (non-overlapping):

![fast_val vs dense sliding windows](docs/_static/img/fast_eval_windows.png)

The windows still tile the whole evaluation span, so metrics remain
representative — they are just computed on disjoint windows instead of every
shifted copy.

---

## Time Series Tasks

This library covers **nine time series tasks** out of the box. Each task has its own experiment class, metrics, and evaluation protocol.

### Forecasting

See [Way 1 — Custom pipeline](#way-1--custom-pipeline-bring-your-own-training-loop) for a complete training and inference example, and [Run Built-In Models](#run-built-in-models) for one-line benchmarking across architectures.

### Probabilistic Forecasting

Any model that can be called multiple times to produce different predictions
(MC Dropout, diffusion, deep ensembles) fits into the probabilistic forecasting
pattern. The full pipeline is: **train → generate N samples → compute quantiles
→ plot / evaluate**.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch_timeseries.dataset import ETTh1
from torch_timeseries.scaler import StandardScaler
from torch_timeseries.dataloader.v2 import (
    ForecastDataModule, WindowConfig, LoaderConfig
)

# ── Step 1: define a model that returns multiple samples ──────────────────────
class MCDropoutForecaster(nn.Module):
    """Linear forecaster with MC Dropout — calling it N times gives N samples."""

    def __init__(self, seq_len: int, pred_len: int, drop: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, 256), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(256, 128),    nn.ReLU(), nn.Dropout(drop),
            nn.Linear(128, pred_len),
        )

    def forward(self, x):                        # x: (B, T, C)
        return self.net(x.transpose(1, 2)).transpose(1, 2)   # (B, pred_len, C)

    def sample(self, x: torch.Tensor, n: int = 200) -> torch.Tensor:
        """Return (B, pred_len, C, n) — dropout stays active for diversity."""
        self.train()
        with torch.no_grad():
            return torch.stack([self(x) for _ in range(n)], dim=-1)

# ── Step 2: load data ─────────────────────────────────────────────────────────
dm = ForecastDataModule(
    dataset=ETTh1(),
    scaler=StandardScaler(),
    window=WindowConfig(window=96, horizon=1, steps=24),
    loader=LoaderConfig(batch_size=64),
)

# ── Step 3: train ─────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = MCDropoutForecaster(seq_len=96, pred_len=24).to(device)
opt    = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(30):
    model.train()
    for batch in dm.train_loader:
        x = batch.x.float().to(device)
        y = batch.y.float().to(device)
        opt.zero_grad()
        nn.MSELoss()(model(x), y).backward()
        opt.step()

# ── Step 4: generate N samples for one validation window ─────────────────────
model.eval()
batch  = next(iter(dm.val_loader))
x_val  = batch.x[:1].float().to(device)   # (1, 96, 7)
y_val  = batch.y[:1].float().to(device)   # (1, 24, 7)

samples = model.sample(x_val, n=200)      # (1, 24, 7, 200)

# ── Step 5: compute prediction intervals from sample quantiles ────────────────
s = samples[0, :, 0, :].cpu().numpy()    # (24, 200) — first feature
lo90, lo50 = np.percentile(s, [5,  25], axis=1)
hi90, hi50 = np.percentile(s, [95, 75], axis=1)
mean_       = s.mean(axis=1)

obs   = x_val[0, :, 0].cpu().numpy()
truth = y_val[0, :, 0].cpu().numpy()
t_obs, t_pred = np.arange(96), np.arange(96, 120)

# ── Step 6: plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(t_obs, obs, color="#888", lw=1.1, label="observed")
ax.plot(t_pred, truth, "--", color="#1f77b4", lw=1.4, label="ground truth")
ax.plot(t_pred, mean_,       color="#d62728", lw=1.4, label="ensemble mean")
ax.fill_between(t_pred, lo90, hi90, alpha=0.18, color="#4C72B0", label="90% PI")
ax.fill_between(t_pred, lo50, hi50, alpha=0.45, color="#4C72B0", label="50% PI")
ax.axvline(96, color="#999", lw=0.8, ls=":")
ax.legend(ncol=2, fontsize=8)
plt.tight_layout()
```

To use the built-in training loop and probabilistic metrics (CRPS, PICP, QICE),
subclass `ProbForecastExp` — `_process_val_batch` must return
`(preds, truths)` where `preds` is `(B, pred_len, C, n_samples)`:

```python
from dataclasses import dataclass
from torch_timeseries.experiments import ProbForecastExp

@dataclass
class MyForecast(ProbForecastExp):
    model_type: str = "MCDropout"

    def _init_model(self):
        self.model = MCDropoutForecaster(self.windows, self.pred_len).to(self.device)

    def _process_train_batch(self, batch):
        x = batch.x.float().to(self.device)
        y = batch.y.float().to(self.device)
        self.model.train()
        return nn.MSELoss()(self.model(x), y)

    def _process_val_batch(self, batch):
        x = batch.x.float().to(self.device)
        y = batch.y.float().to(self.device)
        preds = self.model.sample(x, n=self.num_samples)   # (B, O, C, S)
        return preds, y

result = MyForecast(dataset_type="ETTh1", windows=96, pred_len=24,
                    num_samples=200, device="cuda").run(seed=0)
# -> {'crps': ..., 'picp': ..., 'qice': ..., 'prob_mse': ..., ...}
```

Grey = observed · Blue dashed = ground truth · Red = ensemble mean · Shaded bands = 50 / 90% prediction intervals computed from 200 MC-Dropout samples:

![Probabilistic forecasting with uncertainty bands](docs/_static/img/prob_forecast.png)

### Time Series Generation

`GenerationExp` trains models that learn to *synthesise* new sequences — no forecasting target needed. The training loop feeds sliding windows of the raw series to the model's own loss function; evaluation computes four standard metrics (discriminative score, predictive score, context-FID, correlational score) on generated vs. real sequences.

```python
import torch
import matplotlib.pyplot as plt

from torch_timeseries.model.NsDiff import NsDiff
from torch_timeseries.experiments.NsDiff import NsDiffGeneration

# ── Custom loop: bring your own data ─────────────────────────────────────────
torch.manual_seed(0)
T, C = 96, 3

# build a small synthetic dataset (400 windows, seq_len=96, 3 channels)
real = torch.randn(400, T, C)
ds     = torch.utils.data.TensorDataset(real)
loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

model = NsDiff(seq_len=T, n_features=C, T=100, kernel_size=24)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    for (x,) in loader:
        opt.zero_grad()
        model.loss(x).backward()
        opt.step()

model.eval()
with torch.no_grad():
    samples = model.generate(n=8)   # → (8, 96, 3) on CPU

# ── Experiment runner: built-in Sine / Stocks generation benchmarks ───────────
exp = NsDiffGeneration(
    dataset_type="Sine",   # or "Stocks"
    seq_len=24,
    T=50,
    kernel_size=8,
    epochs=300,
    batch_size=64,
    eval_n_samples=1000,
    device="cuda:0",
)
result = exp.run(seed=1)
# → {'discriminative_score': 0.498, 'predictive_score': 0.012,
#    'context_fid': 0.30, 'correlational_score': 0.22}
print(result)

# ── Plot real vs. generated ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 3), sharey=True)
for i in range(6):
    axes[0].plot(real[i, :, 0].numpy(), color="#aaaaaa", lw=0.8, alpha=0.7)
    axes[1].plot(samples[i, :, 0].numpy(), color="#4C72B0", lw=0.8, alpha=0.7)
axes[0].set_title("Real sequences (channel 0)")
axes[1].set_title("NsDiff generated sequences (channel 0)")
plt.tight_layout()
plt.savefig("nsdiff_generation.png", dpi=120)
```

Grey = real sequences · Blue = NsDiff-generated sequences:

![NsDiff generated vs. real](docs/_static/img/nsdiff_generation.png)

### Imputation

The imputation task randomly masks a fraction of each input window and trains the model to fill in the missing values. Loss is computed only on masked positions. Metrics: MSE, MAE.

```python
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch_timeseries.experiments import ImputationExp

# ── Custom model ──────────────────────────────────────────────────────────────
class LinearImputer(nn.Module):
    """Seq2seq linear model: receives masked input, predicts full window."""
    def __init__(self, seq_len, n_features):
        super().__init__()
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, x):          # x: (B, T, C) — zeros at masked positions
        return self.proj(x.transpose(1, 2)).transpose(1, 2)   # (B, T, C)

# ── Plug into ImputationExp ───────────────────────────────────────────────────
@dataclass
class MyImputation(ImputationExp):
    model_type: str = "LinearImputer"

    def _init_model(self):
        self.model = LinearImputer(
            self.windows, self.dataset.num_features
        ).to(self.device)

    def _process_one_batch(self, batch_masked_x, batch_x, batch_origin_x,
                           batch_mask, batch_x_date_enc):
        batch_masked_x = batch_masked_x.to(self.device, dtype=torch.float32)
        batch_x        = batch_x.to(self.device, dtype=torch.float32)
        return self.model(batch_masked_x), batch_x

result = MyImputation(
    dataset_type="ETTh1", windows=96, mask_rate=0.5,
    epochs=10, device="cuda",
).run(seed=1)
# → {'mse': ..., 'mae': ...}
```

For built-in models use the experiment runner:

```python
from torch_timeseries import Experiment
Experiment(model="DLinear", task="Imputation", dataset="ETTh1",
           windows=96, mask_rate=0.5).run(seeds=[1, 2, 3])
```

Grey = original · Orange = reconstruction · White gaps = masked (50% random):

![Imputation: masked input vs. reconstruction](docs/_static/img/imputation.png)

### Anomaly Detection

Anomaly detection is reconstruction-based: the model is trained to reconstruct normal windows; at test time, high reconstruction error flags anomalies. The per-timestep MSE is used as the anomaly score and thresholded at a configurable percentile. Metrics: precision, recall, F1.

```python
from dataclasses import dataclass
from torch_timeseries.experiments import AnomalyDetectionExp

# ── Custom model (reconstruction) ─────────────────────────────────────────────
class LinearReconstructor(nn.Module):
    def __init__(self, seq_len, n_features):
        super().__init__()
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, x):          # (B, T, C)
        return self.proj(x.transpose(1, 2)).transpose(1, 2)

@dataclass
class MyAnomalyDetection(AnomalyDetectionExp):
    model_type: str = "LinearReconstructor"

    def _init_model(self):
        self.model = LinearReconstructor(
            self.windows, self.dataset.num_features
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_x   # (pred, true)

result = MyAnomalyDetection(
    dataset_type="MSL", windows=100, anomaly_ratio=0.25,
    epochs=10, device="cuda",
).run(seed=1)
# → {'precision': ..., 'recall': ..., 'f1': ...}
```

Built-in models:

```python
Experiment(model="DLinear", task="AnomalyDetection", dataset="MSL",
           windows=100, anomaly_ratio=0.25).run(seeds=[1, 2, 3])
```

Blue = signal · Red shading = detected anomalies · Orange = anomaly score · Dashed = threshold:

![Anomaly detection: reconstruction score and detected regions](docs/_static/img/anomaly_detection.png)

### Classification

Sequence classification on the [UEA Time Series Classification Archive](https://www.timeseriesclassification.com/). The dataset is referenced by its archive name; any UEA dataset downloads automatically. Metrics: accuracy.

```python
from dataclasses import dataclass
from torch_timeseries.experiments import UEAClassificationExp

# ── Custom model (GRU encoder → class logits) ─────────────────────────────────
class GRUClassifier(nn.Module):
    def __init__(self, n_features, n_classes, hidden=64):
        super().__init__()
        self.gru  = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):          # (B, T, C) → (B, n_classes)
        _, h = self.gru(x)
        return self.head(h.squeeze(0))

@dataclass
class MyClassification(UEAClassificationExp):
    model_type: str = "GRUClassifier"

    def _init_model(self):
        self.model = GRUClassifier(
            self.dataset.num_features, self.dataset.num_classes
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_y.long().squeeze(-1)

# windows must match the dataset's fixed sequence length (varies per UEA dataset)
result = MyClassification(
    dataset_type="EthanolConcentration", windows=1751,
    epochs=30, device="cuda",
).run(seed=1)
# → {'accuracy': ...}
```

Built-in models:

```python
Experiment(model="DLinear", task="UEAClassification",
           dataset="EthanolConcentration").run(seeds=[1, 2, 3])
```

Per-class accuracy on EthanolConcentration (4 classes, DLinear):

![Classification: per-class accuracy](docs/_static/img/classification.png)

### Irregular Time Series

Handles asynchronously sampled sequences where observations have arbitrary timestamps and may be missing entirely. Three sub-tasks are supported: **classification**, **interpolation** (reconstruct held-out observations), and **forecasting** (predict future observations after a time-split).

Install optional extras for LatentODE / NeuralCDE / Raindrop:

```bash
pip install "torch-timeseries[irregular]"
```

```python
import torch
from dataclasses import dataclass
from torch_timeseries.dataset.irregular import PhysioNet2012
from torch_timeseries.experiments import IrregularInterpolationExp
from torch_timeseries.model.irregular import mTAN

# ── Interpolation: hold out 20 % of observations; model reconstructs them ────
@dataclass
class mTANInterp(IrregularInterpolationExp):
    model_type: str = "mTAN"
    hidden_size: int = 64
    num_ref_points: int = 16
    num_heads: int = 4

    def _init_model(self):
        self.model = mTAN(
            input_size=self.dm.num_features,
            hidden_size=self.hidden_size,
            num_ref_points=self.num_ref_points,
            num_heads=self.num_heads,
        ).to(self.device)

result = mTANInterp(
    dataset_type="PhysioNet2012",
    query_rate=0.2,
    epochs=30, device="cuda",
).run(seed=1)
# → {'mse': ..., 'mae': ...}
```

Built-in experiment combos:

```python
from torch_timeseries.experiments import (
    mTANIrregularInterpolation,
    GRUDIrregularForecast,
    LatentODEIrregularClassification,
)

# mTAN interpolation on PhysioNet 2012
mTANIrregularInterpolation(dataset_type="PhysioNet2012", epochs=30).run(seed=1)

# GRU-D irregular forecast
GRUDIrregularForecast(dataset_type="PhysioNet2012", obs_frac=0.6, epochs=30).run(seed=1)

# Latent ODE classification (requires torchdiffeq)
LatentODEIrregularClassification(dataset_type="PhysioNet2012", epochs=30).run(seed=1)
```

---

## Development Milestones

### Implemented Datasets

Full list: [Documentation](https://pytorch-timeseries.readthedocs.io/en/latest/modules/dataset.html).

| Datasets | Forecasting | Imputation | Anomaly | Classification |
| -------- | ----------- | ---------- | ------- | -------------- |
| [ETTh1](https://ojs.aaai.org/index.php/AAAI/article/view/17325) | ✅ | ✅ | | |
| [ETTh2](https://ojs.aaai.org/index.php/AAAI/article/view/17325) | ✅ | ✅ | | |
| [ETTm1](https://ojs.aaai.org/index.php/AAAI/article/view/17325) | ✅ | ✅ | | |
| [ETTm2](https://ojs.aaai.org/index.php/AAAI/article/view/17325) | ✅ | ✅ | | |
| [......And More](https://pytorch-timeseries.readthedocs.io/en/latest/modules/dataset.html) | ✅ | ✅ | ✅ | ✅ |

### Implemented Tasks

- [x] Forecast
- [x] Probabilistic Forecast
- [x] Imputation
- [x] Anomaly Detection
- [x] Classification (UEA datasets)
- [x] Generation
- [x] Irregular Classification
- [x] Irregular Interpolation
- [x] Irregular Forecasting
- [ ] Contribute your own task!

### Implemented Models

#### Regular time series

| Models | Forecasting | Imputation | Anomaly | Classification |
| ------ | ----------- | ---------- | ------- | -------------- |
| [Informer (2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17325) | ✅ | ✅ | ✅ | ✅ |
| [Autoformer (2021)](https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html) | ✅ | ✅ | ✅ | ✅ |
| [FEDformer (2022)](https://proceedings.mlr.press/v162/zhou22g.html) | ✅ | ✅ | ✅ | ✅ |
| [DLinear (2022)](https://ojs.aaai.org/index.php/AAAI/article/view/26317) | ✅ | ✅ | ✅ | ✅ |
| [PatchTST (2022)](https://openreview.net/forum?id=Jbdc0vTOcol) | ✅ | ✅ | ✅ | ✅ |
| [iTransformer (2024)](https://openreview.net/forum?id=JePfAI8fah) | ✅ | ✅ | ✅ | ✅ |
| [SegRNN (2024)](https://openreview.net/forum?id=jeqE7rqz2L) | ✅ | ✅ | ✅ | ✅ |
| [TimeMixer (2024)](https://openreview.net/forum?id=7oLshfEIC2) | ✅ | ✅ | ✅ | ✅ |
| [TiDE (2023)](https://arxiv.org/abs/2304.08424) | ✅ | ✅ | ✅ | ✅ |
| [N-HiTS (2023)](https://ojs.aaai.org/index.php/AAAI/article/view/26253) | ✅ | ✅ | ✅ | ✅ |

#### Irregular time series

| Models | Irr. Classify | Interpolation | Irr. Forecast | Extra deps |
| ------ | ------------- | ------------- | ------------- | ---------- |
| [GRU-D (2018)](https://www.nature.com/articles/s41598-018-24271-9) | ✅ | ✅ | ✅ | — |
| [mTAN (2021)](https://openreview.net/forum?id=4c0J6lwQ4_) | ✅ | ✅ | ✅ | — |
| [LatentODE (2019)](https://proceedings.neurips.cc/paper/2019/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html) | ✅ | ✅ | ✅ | torchdiffeq |
| [NeuralCDE (2020)](https://proceedings.neurips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html) | ✅ | | | torchcde |
| [Raindrop (2022)](https://openreview.net/forum?id=Kwm8I7dU-l5) | ✅ | | | torch_geometric |



## Dev Install

> This library assumes PyTorch is already installed: https://pytorch.org/get-started/locally/
>
> Recommended Python: 3.8.1+

```bash
# 1. fork and clone
git clone https://github.com/wayne155/pytorch_timeseries

# 2. install dependencies
pip install -r ./requirements.txt

# 3. make changes and open a pull request
```
