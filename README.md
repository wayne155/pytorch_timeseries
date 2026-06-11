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

An all-in-one deep learning library for time series research.
[Full documentation](https://pytorch-timeseries.readthedocs.io/en/latest/).

- Datasets downloaded automatically
- Easy to extend with your own model
- Highly customizable pipeline
- One-command experiment runner



## Installation

```bash
pip install torch-timeseries
```

> **Python 3.8+ required.**



## Two Ways to Use

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

Use this pattern when you need a non-standard training loop, custom loss, or are prototyping a new architecture.

#### Custom Datasets

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

#### Fast evaluation windows (`fast_val` / `fast_test`)

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

The windows still tile the whole evaluation span, so metrics remain
representative — they are just computed on disjoint windows instead of every
shifted copy.

#### Probabilistic forecasting

`ProbForecastExp` adapts the default training paradigm for sample-based
probabilistic models (diffusion, flows, deep ensembles, ...). It differs from
point forecasting in three ways:

1. **Training** — the model computes its own loss (ELBO, diffusion loss, ...),
   so `_process_train_batch(batch)` returns the loss directly instead of an
   `(x, y)` pair.
2. **Inference** — `_process_val_batch(batch)` returns an ensemble
   `preds` of shape `(batch, pred_len, num_features, num_samples)` plus
   `truths` of shape `(batch, pred_len, num_features)`.
3. **Evaluation** — val/test use `fast_val`/`fast_test` non-overlapping
   windows by default, and metrics are the probabilistic suite from
   `torch_timeseries.metrics`: `CRPS`, `CRPSSum`, `QICE`, `PICP`, plus
   `ProbMSE`/`ProbMAE`/`ProbRMSE` on the ensemble mean. Early stopping
   monitors validation CRPS.

```python
from dataclasses import dataclass
import torch
from torch_timeseries.experiments import ProbForecastExp

@dataclass
class MyDiffusionForecast(ProbForecastExp):
    model_type: str = "MyDiffusion"

    def _init_model(self):
        self.model = MyDiffusion(...).to(self.device)

    def _process_train_batch(self, batch):
        x = batch.x.to(self.device).float()
        y = batch.y.to(self.device).float()
        return self.model.loss(x, y)            # model returns its own loss

    def _process_val_batch(self, batch):
        x = batch.x.to(self.device).float()
        y = batch.y.to(self.device).float()
        preds = self.model.sample(x, n=self.num_samples)  # (B, O, N, S)
        return preds, y

exp = MyDiffusionForecast(dataset_type="ETTh1", windows=96, pred_len=96,
                          num_samples=100, device="cuda:0")
exp.run(seed=0)   # -> {'crps': ..., 'crps_sum': ..., 'picp': ..., 'qice': ..., ...}
```

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

Use this pattern when you want to benchmark on standard tasks without writing boilerplate.



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
- [x] Imputation
- [x] Anomaly Detection
- [x] Classification (UEA datasets)
- [ ] Contribute your own task!

### Implemented Models

| Models | Forecasting | Imputation | Anomaly | Classification |
| ------ | ----------- | ---------- | ------- | -------------- |
| [Informer (2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17325) | ✅ | ✅ | ✅ | ✅ |
| [Autoformer (2021)](https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html) | ✅ | ✅ | ✅ | ✅ |
| [FEDformer (2022)](https://proceedings.mlr.press/v162/zhou22g.html) | ✅ | ✅ | ✅ | ✅ |
| [DLinear (2022)](https://ojs.aaai.org/index.php/AAAI/article/view/26317) | ✅ | ✅ | ✅ | ✅ |
| [PatchTST (2022)](https://openreview.net/forum?id=Jbdc0vTOcol) | ✅ | ✅ | ✅ | ✅ |
| [iTransformer (2024)](https://openreview.net/forum?id=JePfAI8fah) | ✅ | ✅ | ✅ | ✅ |



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
