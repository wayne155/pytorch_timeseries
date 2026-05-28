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
from torch_timeseries.dataset import ETTh1
from torch_timeseries.scaler import StandardScaler
from torch_timeseries.dataloader.v2 import (
    ForecastDataModule, WindowConfig, SplitConfig, LoaderConfig
)

# dataset is downloaded automatically on first use
dataset = ETTh1("./data")

dm = ForecastDataModule(
    dataset=dataset,
    scaler=StandardScaler(),
    window=WindowConfig(window=96, horizon=1, steps=96),
    split=SplitConfig(train=0.7, val=0.1, test=0.2),
    loader=LoaderConfig(batch_size=32),
)

# each batch is a TSBatch: .x, .y, .x_time, .y_time, .x_raw, .y_raw
for batch in dm.train_loader:
    x = batch.x.float()        # (B, 96, num_features)
    y = batch.y.float()        # (B, 96, num_features)
    # ... your model, loss, optimizer here
```

Use this pattern when you need a non-standard training loop, custom loss, or are prototyping a new architecture.

---

### Way 2 — Default experiments (one command, results saved automatically)

Use the built-in experiment runner. Pick a model, task, and dataset — the library handles data loading, training, evaluation, and result saving.

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
```

**Python API:**

```python
from torch_timeseries.experiments import DLinearForecast

exp = DLinearForecast(
    dataset_type="ETTh1",
    windows=96,
    pred_len=96,
    lr=0.001,
)
exp.run(3)          # run with seed=3
exp.runs([1, 2, 3]) # run with multiple seeds
```

Use this pattern when you want to benchmark on standard tasks without writing boilerplate.

### Leaderboard artifacts

Generate static Markdown, CSV, and JSON leaderboard files from local experiment
results plus curated YAML entries:

```bash
pytexp leaderboard \
  --results_dir ./results \
  --entries_dir leaderboard/entries \
  --output_dir results/leaderboard \
  --docs_dir docs/leaderboard
```



## Custom Models

To plug in your own model, subclass the task experiment class, define `_init_model`, and register it:

```python
from dataclasses import dataclass
from torch_timeseries.experiments import ForecastExp
from torch_timeseries import register_model
import torch.nn as nn

@dataclass
class MyModelParameters:
    hidden_dim: int = 64

@dataclass
class MyModelForecast(ForecastExp, MyModelParameters):
    model_type: str = "MyModel"

    def _init_model(self):
        self.model = MyNet(
            seq_len=self.windows,
            pred_len=self.pred_len,
            hidden_dim=self.hidden_dim,
        )
        self.model = self.model.to(self.device)

register_model(MyModelForecast)
```

Then run it with any supported task and dataset:

```bash
pytexp --model MyModel --task Forecast --dataset_type ETTh1 run 1
```



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
