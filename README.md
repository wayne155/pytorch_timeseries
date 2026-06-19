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
- [Forecaster — High-Level API](#forecaster--high-level-api)
  - [Fit · Predict · Score](#fit--predict--score)
  - [Model Comparison](#model-comparison)
  - [Uncertainty Quantification](#uncertainty-quantification)
  - [Anomaly Detection](#anomaly-detection-1)
  - [Signal Analysis](#signal-analysis)
  - [Explainability](#explainability)
  - [Transfer Learning](#transfer-learning)
  - [Ensemble & Composition](#ensemble--composition)
  - [Deployment Utilities](#deployment-utilities)
  - [Change Point Detection](#change-point-detection)
  - [Granger Causality](#granger-causality)
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
- [Data Loading Reference](#data-loading-reference)
  - [ForecastDataModule](#forecastdatamodule)
  - [WindowConfig](#windowconfig)
  - [LoaderConfig](#loaderconfig)
  - [SplitConfig](#splitconfig)
  - [Scalers](#scalers)
  - [Other DataModules](#other-datamodules)
- [Development Milestones](#development-milestones)
  - [Implemented Datasets](#implemented-datasets)
  - [Implemented Tasks](#implemented-tasks)
  - [Implemented Models — Point Forecasters](#implemented-models--point-forecasters)
  - [Implemented Models — Probabilistic Forecasters](#implemented-models--probabilistic-forecasters)
  - [Implemented Models — Generation](#implemented-models--generation)
  - [Implemented Models — Irregular Time Series](#implemented-models--irregular-time-series)
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

---

## Forecaster — High-Level API

`Forecaster` is a scikit-learn-style wrapper that handles training, evaluation,
uncertainty, explainability, and deployment for any of the 80+ built-in models
— no boilerplate required.

```python
import numpy as np
from torch_timeseries import Forecaster

# Synthetic multivariate series: 2 000 timesteps × 7 channels
rng = np.random.default_rng(0)
X   = rng.standard_normal((2_000, 7)).astype("float32")
```

---

### Fit · Predict · Score

```python
fc = Forecaster(
    "iTransformer",          # any registered model name
    seq_len  = 96,           # look-back window
    pred_len = 24,           # forecast horizon
    epochs   = 20,
    lr       = 1e-3,
    patience = 5,            # early stopping
    normalize= True,
    verbose  = True,
)

fc.fit(X)                    # train / val split handled automatically

# Forecast the next 24 steps from the last 96
ctx  = X[-96:]               # (96, 7)
pred = fc.predict(ctx)       # (24, 7)

# Labeled DataFrame — channel names optional
df = fc.forecast_dataframe(
    ctx,
    channel_names=["OT", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
    start_index=2_000,
)
print(df.head())
#       OT    HUFL    HULL    MUFL    MULL    LUFL    LULL
# 2000  0.12  -0.33   0.87   0.04   -0.21   0.56   -0.08
# ...

# Point metrics
print(fc.score(X))
# {'MSE': 0.981, 'MAE': 0.789, 'RMSE': 0.990, 'SMAPE': 98.3}

# Per-channel breakdown
per_ch = fc.score_per_channel(X)
print(per_ch["MAE"])         # np.ndarray of shape (7,)

# Descriptive statistics of the input
print(fc.describe(X, channel_names=["OT","HUFL","HULL","MUFL","MULL","LUFL","LULL"]))
#                      OT      HUFL   ...
# count          2000.00   2000.00   ...
# mean              0.00      0.01   ...
# autocorr_lag1    -0.02      0.00   ...
```

---

### Model Comparison

```python
from torch_timeseries import compare, compare_plot

# Benchmark five models on the same train / test split
X_train, X_test = X[:1_500], X[1_500:]

results = compare(
    models = ["DLinear", "NLinear", "PatchTST", "iTransformer", "TimeMixer"],
    X_train = X_train,
    X_test  = X_test,
    seq_len = 96,
    pred_len= 24,
    epochs  = 10,
    verbose = False,
)
# {'DLinear': {'MSE': 0.97, 'MAE': 0.78, ...}, 'PatchTST': {...}, ...}

# Sorted bar chart — best model on the left
fig = compare_plot(results, metric="MAE", title="Model comparison — 7-channel synthetic")
fig.savefig("model_comparison.png", dpi=150)

# Or from an existing fitted Forecaster, inheriting its hyperparams:
report = fc.compare_models(
    models  = ["DLinear", "NLinear", "TimeMixer"],
    X_train = X_train,
    X_test  = X_test,
)

# Compare against the naive persistence baseline in one call
bvp = fc.score_vs_persistence(X_test)
print(f"Model MAE : {bvp['model']['MAE']:.4f}")
print(f"Persist.  : {bvp['persistence']['MAE']:.4f}")
```

---

### Uncertainty Quantification

Three complementary approaches in a single API:

```python
# ── 1. MC-Dropout uncertainty ────────────────────────────────────────────────
unc = fc.predict_uncertainty(ctx, n_samples=200)
# {'mean': (24,7), 'std': (24,7), 'lower': (24,7), 'upper': (24,7)}

# ── 2. Post-hoc calibration — adjust interval width to hit target coverage ──
fc.calibrate(X_train, target_coverage=0.90, n_samples=200)
# Now predict_uncertainty() applies the learned scale automatically.
unc_cal = fc.predict_uncertainty(ctx, n_samples=200)

# ── 3. Conformal prediction intervals — model-free, guaranteed coverage ──────
intervals = fc.predict_interval(ctx, X_cal=X_train, coverage=0.90)
# {'lower': (24,7), 'upper': (24,7)}

# ── Visualise uncertainty on channel 0 ───────────────────────────────────────
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 3.5))
t_ctx  = np.arange(96)
t_pred = np.arange(96, 120)

ax.plot(t_ctx, ctx[:, 0], color="#888", lw=1, label="context")
ax.plot(t_pred, pred[:, 0], color="#d62728", lw=1.5, label="forecast")
ax.fill_between(t_pred,
    intervals["lower"][:, 0], intervals["upper"][:, 0],
    alpha=0.25, color="#4C72B0", label="90% conformal PI")
ax.fill_between(t_pred,
    unc_cal["lower"][:, 0], unc_cal["upper"][:, 0],
    alpha=0.20, color="#ff7f0e", label="90% calibrated MC-Dropout")
ax.axvline(96, color="#bbb", lw=0.8, ls=":")
ax.legend(ncol=2, fontsize=8)
ax.set_title("Forecast with uncertainty — OT channel")
plt.tight_layout()
fig.savefig("uncertainty.png", dpi=150)

# ── Stochastic scenario fan chart ────────────────────────────────────────────
mc = fc.montecarlo_forecast(ctx, steps=48, n_scenarios=300)
fig2 = fc.plot_scenarios(mc, channel=0, X_context=ctx, n_scenarios_to_plot=50)
fig2.savefig("scenarios.png", dpi=150)
```

---

### Anomaly Detection

```python
# Rolling residual-based scores
scores, indices = fc.anomaly_score(X_test, stride=1, reduction="mean")

# Flag the top 3 % as anomalous
anom_df = fc.flag_anomalies(X_test, contamination=0.03)
print(anom_df[anom_df["anomaly"]].head())
#    timestep     score  anomaly
# 7        7  2.341876     True
# 42      42  1.984321     True

# Inject a spike and verify detection
X_spike = X_test.copy()
X_spike[50:53, :] += 5.0          # abrupt spike in all channels
anom_spike = fc.flag_anomalies(X_spike, contamination=0.05)

# Visualise
fig = fc.plot_anomalies(
    X_spike, anom_spike,
    channel=0,
    title="Anomaly detection — injected spike at t=50–52",
)
fig.savefig("anomalies.png", dpi=150)
```

---

### Signal Analysis

All analysis methods are **static** — no model fitting required.

```python
# ── Autocorrelation ──────────────────────────────────────────────────────────
lags, acf = Forecaster.autocorrelation(X, max_lag=48, channel=0)
fig = Forecaster.plot_acf(X, max_lag=48, title="ACF — OT channel")
fig.savefig("acf.png", dpi=150)

# ── Power spectrum ───────────────────────────────────────────────────────────
freqs, psd = Forecaster.spectral_density(X, channel=0)
fig = Forecaster.plot_spectral_density(X, title="Power spectral density")
fig.savefig("psd.png", dpi=150)

# ── Seasonal decomposition ───────────────────────────────────────────────────
decomp = Forecaster.seasonal_decompose(X, period=24, method="additive")
# decomp keys: 'original', 'trend', 'seasonal', 'residual'

fig = Forecaster.plot_decomposition(decomp, channel=0, title="Additive decomp (period=24)")
fig.savefig("decomposition.png", dpi=150)

# ── Missing-value imputation (before fitting) ─────────────────────────────────
X_dirty = X.copy()
X_dirty[100:105, 0] = np.nan          # introduce NaN
X_clean = Forecaster.interpolate_missing(X_dirty, method="linear")

# ── Horizon-level error profile ───────────────────────────────────────────────
profile = fc.horizon_error_profile(X_test, metric="MAE")
print(profile)   # (24,) — error grows with horizon

fig = fc.plot_horizon_error_profile(X_test, metric="MAE",
                                    title="How error grows with forecast step")
fig.savefig("horizon_error.png", dpi=150)

# ── Channel correlation heatmap ───────────────────────────────────────────────
corr = Forecaster.channel_correlation(X)   # (7, 7) Pearson matrix
fig  = Forecaster.plot_channel_correlation(
    X,
    channel_names=["OT","HUFL","HULL","MUFL","MULL","LUFL","LULL"],
    title="Channel Pearson correlation",
)
fig.savefig("channel_corr.png", dpi=150)

# ── Rolling correlation between two channels ──────────────────────────────────
ts, rc = Forecaster.rolling_correlation(X, window=48, channel_i=0, channel_j=1)
fig = Forecaster.plot_rolling_correlation(X, window=48, channel_i=0, channel_j=1,
                                          title="Rolling r(OT, HUFL) — window=48")
fig.savefig("rolling_corr.png", dpi=150)

# ── Q-Q plot: are residuals Gaussian? ────────────────────────────────────────
dist = fc.residual_distribution(X_test, channel=0)
print(f"bias={dist['mean']:.4f}  skew={dist['skewness']:.4f}  kurt={dist['kurtosis']:.4f}")

fig = fc.plot_qq(X_test, channel=0, title="Residual Q-Q (should follow y = x for Gaussian)")
fig.savefig("qq.png", dpi=150)

# ── Descriptive stats DataFrame ───────────────────────────────────────────────
stats = Forecaster.describe(X, channel_names=["OT","HUFL","HULL","MUFL","MULL","LUFL","LULL"])
print(stats.T)   # transpose for readability
```

---

### Explainability

```python
# ── Gradient saliency: which input timesteps drive step-1 of channel OT? ────
grad = fc.input_gradient(
    X_test[:96],            # context window
    target_step    = 0,     # first forecast step
    target_channel = 0,     # OT channel
    absolute       = True,  # magnitude only
)
# grad: (96, 7) — large values = high influence on the forecast

fig = fc.plot_saliency(
    X_test[:96],
    target_step    = 0,
    target_channel = 0,
    channel_names  = ["OT","HUFL","HULL","MUFL","MULL","LUFL","LULL"],
    title          = "Input saliency → OT forecast step 1",
)
fig.savefig("saliency.png", dpi=150)

# ── Permutation importance: which channel hurts most when shuffled? ──────────
perm = fc.feature_importance(X_test, metric="MAE", n_repeats=10)
# {'ch0': 0.12, 'ch1': 0.04, ...}

# ── Timestep importance: which context lag matters? ──────────────────────────
ts_imp = fc.timestep_importance(X_test, metric="MAE", n_repeats=5)
fig    = fc.plot_timestep_importance(X_test, title="Lag importance")
fig.savefig("timestep_importance.png", dpi=150)

# ── Bias-variance decomposition via bootstrap ─────────────────────────────────
bvd = fc.error_decomposition(X_train, X_test, n_bootstrap=20, seed=0)
print(f"bias²={bvd['bias2']:.4f}   variance={bvd['variance']:.4f}   total={bvd['total_mse']:.4f}")
```

---

### Transfer Learning

```python
# ── Copy weights from a pretrained model, then fine-tune ─────────────────────
fc_source = Forecaster("iTransformer", seq_len=96, pred_len=24, epochs=30)
fc_source.fit(X_train)

# Target domain — new dataset, same architecture
fc_target = Forecaster("iTransformer", seq_len=96, pred_len=24, epochs=5)
fc_target.fit(X_test[:500])         # brief fit to initialise architecture

fc_target.copy_weights_from(fc_source)   # transplant pretrained weights
fc_target.freeze_layers(["embedding"])   # lock input projections
fc_target.partial_fit(X_test[:500])      # fine-tune on target domain

print(f"Frozen params : {fc_target.frozen_parameter_count():,}")
print(f"Total params  : {fc_target.count_parameters()['total']:,}")

# ── Save / reload / move to a different device ────────────────────────────────
fc_source.save("./checkpoints/iTransformer_source")
fc_reload = Forecaster.from_pretrained(
    "./checkpoints/iTransformer_source",
    device = "cpu",          # override device at load time
)

# ── Export to ONNX for production serving ────────────────────────────────────
fc_source.to_onnx("./iTransformer.onnx")   # (seq_len, C) → (pred_len, C)
```

---

### Ensemble & Composition

```python
from torch_timeseries import (
    EnsembleForecaster, StackedForecaster,
    MultiChannelForecaster, Pipeline,
)

# ── Weighted ensemble of heterogeneous models ─────────────────────────────────
ens = EnsembleForecaster(
    forecasters = [
        ("dlinear",      Forecaster("DLinear",      seq_len=96, pred_len=24, epochs=10)),
        ("patchtst",     Forecaster("PatchTST",     seq_len=96, pred_len=24, epochs=10)),
        ("itransformer", Forecaster("iTransformer", seq_len=96, pred_len=24, epochs=10)),
    ],
    weights = [0.2, 0.4, 0.4],          # custom weights; None = uniform
)
ens.fit(X_train)

pred_ens = ens.predict(X_test[:96])     # (24, 7)
unc_ens  = ens.predict_std(X_test[:96]) # {'mean': ..., 'std': ...}

# ── Stacked (boosted) forecaster ──────────────────────────────────────────────
stacked = StackedForecaster(
    base = Forecaster("DLinear",  seq_len=96, pred_len=24, epochs=10),
    meta = Forecaster("NLinear",  seq_len=96, pred_len=24, epochs=5),
)
stacked.fit(X_train)

# ── One independent model per channel (channel-independent mode at API level) ─
mc = MultiChannelForecaster(
    base = Forecaster("DLinear", seq_len=96, pred_len=24, epochs=10)
)
mc.fit(X_train)
pred_mc = mc.predict(X_test[:96])       # (24, 7)

# ── Preprocessing pipeline ────────────────────────────────────────────────────
def log1p(X):   return np.log1p(np.abs(X)) * np.sign(X)
def expm1(X):   return np.expm1(np.abs(X)) * np.sign(X)

pipe = Pipeline(preprocessor=log1p,
                forecaster=Forecaster("DLinear", seq_len=96, pred_len=24, epochs=10))
pipe.set_inverse(expm1)
pipe.fit(X_train)
pred_pipe = pipe.predict(X_test[:96])   # automatically un-transforms output

# ── sklearn-compatible wrapper (GridSearchCV / Pipeline ready) ────────────────
from torch_timeseries import SklearnForecaster

sk = SklearnForecaster(
    model    = "DLinear",
    seq_len  = 96,
    pred_len = 24,
    epochs   = 5,
)
sk.fit_ts(X_train)           # accepts (T, C) directly
sk.score(X_test[:96])        # returns negative MSE (sklearn convention)
```

---

### Deployment Utilities

```python
# ── Latency / throughput profiling ────────────────────────────────────────────
stats = fc.profile(X_test[:96], n_repeats=100)
print(f"mean latency : {stats['mean_ms']:.2f} ms")
print(f"throughput   : {stats['throughput']:.0f} windows/s")

# ── Move to GPU / CPU at runtime ──────────────────────────────────────────────
fc.set_device("cuda:0")
pred_gpu = fc.predict(X_test[:96])
fc.set_device("cpu")

# ── Warm up JIT / CUDA caches before timing ───────────────────────────────────
fc.warmup(n=5)

# ── Memory-efficient rolling prediction on a long series ─────────────────────
preds_all = fc.chunked_predict(X_test, chunk_size=64)
# shape: (n_windows, pred_len, C)

# ── Export rolling forecasts to CSV ──────────────────────────────────────────
fc.export_predictions(
    X_test,
    path          = "./forecasts.csv",
    channel_names = ["OT","HUFL","HULL","MUFL","MULL","LUFL","LULL"],
)

# ── Stream one step at a time (real-time inference) ───────────────────────────
for pred_step in fc.stream_predict(X_test):
    pass  # pred_step: (24, 7) — updates as each new timestep arrives

# ── Wrap dataset for custom training loops ────────────────────────────────────
import torch
ds     = fc.to_torch_dataset(X_train)          # WindowDataset
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
x, y   = next(iter(loader))                    # (32, 96, 7), (32, 24, 7)

# ── Hyperparameter search ─────────────────────────────────────────────────────
best = fc.tune(
    X_train,
    param_grid = {"lr": [1e-3, 5e-4], "d_model": [64, 128], "n_heads": [4, 8]},
    n_iter     = 8,
    n_splits   = 3,
)
print(best)    # {'lr': 0.0005, 'd_model': 128, 'n_heads': 8, 'val_loss': 0.83}
```

---

### Change Point Detection

```python
# ── Abrupt shift in mean ──────────────────────────────────────────────────────
X_shift = np.vstack([
    rng.standard_normal((500, 7)),
    rng.standard_normal((500, 7)) + 3.0,   # level shift at t=500
    rng.standard_normal((500, 7)),
])

cps = Forecaster.detect_change_points(X_shift, window=40, channel=0)
print(cps)    # [490, 510, 998, ...]  — detected near t=500 and t=1000

fig = Forecaster.plot_change_points(
    X_shift, cps,
    channel = 0,
    title   = "Change point detection — two abrupt level shifts",
)
fig.savefig("change_points.png", dpi=150)
```

---

### Granger Causality

```python
# Build a series where channel 0 Granger-causes channel 1 with lag 3
T = 1_000
x0 = rng.standard_normal(T)
x1 = np.roll(x0, 3) + 0.3 * rng.standard_normal(T)   # x1[t] ≈ x0[t-3]
x2 = rng.standard_normal(T)                            # independent noise
X_gc = np.column_stack([x0, x1, x2]).astype("float32")

# Pairwise F-statistics — high F[i,j] = channel i Granger-causes channel j
f_mat = Forecaster.granger_test(X_gc, max_lag=5)
print(f_mat.round(1))
# [[  0.   89.3   0.4]
#  [  0.3   0.    0.3]
#  [  0.5   0.8   0. ]]   ← F[0,1] >> everything else ✓

fig = Forecaster.plot_granger(
    X_gc,
    max_lag       = 5,
    channel_names = ["x0", "x1", "x2"],
    title         = "Granger causality — x0 → x1 with lag 3",
)
fig.savefig("granger.png", dpi=150)
```

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

## Data Loading Reference

The library's data pipeline is built around three composable config objects and a
family of `DataModule` classes — one per task type.

### ForecastDataModule

```python
from torch_timeseries.dataloader.v2 import (
    ForecastDataModule, WindowConfig, LoaderConfig, SplitConfig
)
from torch_timeseries.dataset import ETTh1
from torch_timeseries.scaler import StandardScaler

dm = ForecastDataModule(
    dataset = ETTh1(),
    scaler  = StandardScaler(),
    window  = WindowConfig(window=96, horizon=1, steps=24),
    split   = SplitConfig(train=0.7, val=0.1, test=0.2),   # optional
    loader  = LoaderConfig(batch_size=64, num_workers=4),   # optional
)

# three ready-to-use PyTorch DataLoaders
dm.train_loader   # shuffled training windows
dm.val_loader     # non-shuffled validation windows
dm.test_loader    # non-shuffled test windows

# convenience attributes
dm.num_features   # == dataset.num_features (int)
dm.scaler         # fitted scaler instance
```

Each batch is a `Batch` object with fields:

| Field | Shape | Description |
|-------|-------|-------------|
| `batch.x` | `(B, window, C)` | Input context window |
| `batch.y` | `(B, steps, C)` | Target prediction window |
| `batch.x_date_enc` | `(B, window, date_features)` | Time encoding for input |
| `batch.y_date_enc` | `(B, steps, date_features)` | Time encoding for target |

```python
for batch in dm.train_loader:
    x = batch.x.float().to(device)          # (64, 96, 7)
    y = batch.y.float().to(device)          # (64, 24, 7)
    x_enc = batch.x_date_enc.float().to(device)  # (64, 96, 4)
```

---

### WindowConfig

Controls how windows are cut from the time series.

```python
WindowConfig(
    window      = 96,     # look-back length (seq_len)
    horizon     = 1,      # gap between context end and prediction start
    steps       = 24,     # prediction length (pred_len)
    stride      = 1,      # sliding step during training (1 = dense)
    fast_val    = False,  # non-overlapping windows at validation (faster)
    fast_test   = False,  # non-overlapping windows at test (faster)
    input_columns  = None,  # list of column names to use as input features
    target_columns = None,  # list of column names to predict (default=all)
    time_enc_cfg   = TimeEncConfig(),  # time encoding configuration
)
```

**`fast_val` / `fast_test`** — switch evaluation splits to non-overlapping
windows (stride = `window + horizon + steps − 1`).  Training always uses the
dense stride.  This dramatically reduces inference calls for expensive models
(diffusion, neural ODEs) while keeping metrics representative.

---

### LoaderConfig

Controls PyTorch `DataLoader` settings.

```python
LoaderConfig(
    batch_size    = 32,     # samples per mini-batch
    num_workers   = 0,      # parallel data-loading workers
    shuffle_train = True,   # shuffle training set each epoch
    pin_memory    = False,  # pin memory for CUDA transfers
)
```

---

### SplitConfig

Controls train / val / test proportions.  Ratios must sum to ≤ 1.

```python
from torch_timeseries.dataloader.v2 import SplitConfig

SplitConfig(train=0.7, val=0.1, test=0.2)   # default: 70/10/20
```

Many built-in datasets have a canonical benchmark split that is applied
automatically when no `SplitConfig` is given (e.g. ETT series use the
standard 12/4/4 months split used in the Informer paper).

---

### Scalers

```python
from torch_timeseries.scaler import StandardScaler, MinMaxScaler

StandardScaler()   # zero-mean unit-variance normalisation (default)
MinMaxScaler()     # scale to [0, 1] per channel
```

The scaler is fitted on the training split only and applied to all splits —
no data leakage.

---

### Other DataModules

| DataModule | Task | Import path |
|-----------|------|-------------|
| `ForecastDataModule` | Forecasting | `torch_timeseries.dataloader.v2` |
| `ImputationDataModule` | Imputation | `torch_timeseries.dataloader.v2` |
| `AnomalyDataModule` | Anomaly detection | `torch_timeseries.dataloader.v2` |
| `UEADataModule` | UEA classification | `torch_timeseries.dataloader.v2` |
| `GenerationDataModule` | Time series generation | `torch_timeseries.dataloader.v2` |
| `IrregularForecastDataModule` | Irregular forecasting | `torch_timeseries.dataloader.v2` |
| `IrregularInterpolationDataModule` | Irregular interpolation | `torch_timeseries.dataloader.v2` |
| `IrregularClassificationDataModule` | Irregular classification | `torch_timeseries.dataloader.v2` |

Each task-specific DataModule exposes the same `train_loader` / `val_loader` /
`test_loader` interface; only the batch layout differs between tasks.

**Imputation batch** — `batch.masked_x` (masked input), `batch.x` (target), `batch.mask` (bool tensor)

**Anomaly batch** — `batch.x` (window), `batch.origin_x` (original before augmentation)

**UEA batch** — `batch.x` (sequence), `batch.y` (class label), `batch.padding_masks`

---

## Development Milestones

### Implemented Datasets

All datasets download automatically on first use to `~/.torchtimeseries/data/`.

#### Forecasting / Imputation datasets

| Dataset | Freq | Features | Length | Source |
|---------|------|----------|--------|--------|
| ETTh1 | hourly | 7 | 17,420 | [Informer (AAAI 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17325) |
| ETTh2 | hourly | 7 | 17,420 | [Informer (AAAI 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17325) |
| ETTm1 | 15-min | 7 | 69,680 | [Informer (AAAI 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17325) |
| ETTm2 | 15-min | 7 | 69,680 | [Informer (AAAI 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17325) |
| Weather | hourly | 21 | 52,696 | Max Planck Institute |
| Traffic | hourly | 862 | 17,544 | Caltrans PeMS |
| Electricity | 15-min | 321 | 26,304 | UCI |
| Exchange Rate | daily | 8 | 7,588 | LSTNet |
| ILI | weekly | 7 | 966 | CDC |
| Solar Energy | hourly | 137 | 52,560 | NREL |
| M4 | mixed | 1 | varies | M4 competition |
| Stocks | daily | varies | varies | Yahoo Finance |

#### Anomaly detection datasets

| Dataset | Description |
|---------|-------------|
| MSL | NASA Mars Science Laboratory telemetry |
| SMAP | NASA Soil Moisture Active Passive satellite |
| SMD | Server Machine Dataset (OmniAnomaly) |
| PSM | Pooled Server Metrics (eBay) |
| SWaT | Secure Water Treatment (SUTD) |

#### Classification datasets

| Dataset | Description |
|---------|-------------|
| UEA archive | All 30+ UEA Time Series Classification datasets auto-download by name |

#### Irregular time series datasets

| Dataset | Description |
|---------|-------------|
| PhysioNet 2012 | ICU patient records (48-h, 35 variables, irregular sampling) |

#### Synthetic / Simulation datasets

| Dataset | Description |
|---------|-------------|
| Sine | Single-frequency sinusoids (configurable freq, phase, noise) |
| SimFreq | Multi-frequency synthetic series |
| SimFreqCF | Cross-frequency coupled synthetic series |

---

### Implemented Tasks

- [x] Forecasting
- [x] Probabilistic Forecasting
- [x] Imputation
- [x] Anomaly Detection
- [x] Classification (UEA datasets)
- [x] Generation
- [x] Irregular Classification
- [x] Irregular Interpolation
- [x] Irregular Forecasting
- [ ] Contribute your own task!

---

### Implemented Models — Point Forecasters

All point forecasters accept `(B, T, C)` input and return `(B, pred_len, C)`.
They are usable both through the **Forecaster** high-level API and the low-level
experiment runner.

#### Transformer family

| Model | Key Idea | Paper |
|-------|----------|-------|
| VanillaTransformer | Baseline encoder-decoder transformer | [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) |
| Informer | ProbSparse attention, distilling | [Zhou et al., AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17325) |
| Autoformer | Auto-Correlation + decomposition | [Wu et al., NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html) |
| FEDformer | Frequency-enhanced decomposed transformer | [Zhou et al., ICML 2022](https://proceedings.mlr.press/v162/zhou22g.html) |
| NSTransformer | Non-stationary attention | [Liu et al., NeurIPS 2022](https://arxiv.org/abs/2205.14415) |
| PatchTST | Patch tokenisation, channel independence | [Nie et al., ICLR 2023](https://openreview.net/forum?id=Jbdc0vTOcol) |
| Crossformer | Cross-dimension attention | [Zhang & Yan, ICLR 2023](https://openreview.net/forum?id=vSVLM2j9eie) |
| iTransformer | Inverted attention (channels as tokens) | [Liu et al., ICLR 2024](https://openreview.net/forum?id=JePfAI8fah) |
| Pathformer | Multi-scale path attention | [Chen et al., ICLR 2024](https://openreview.net/forum?id=lJkOCMP2aW) |
| ETSformer | Exponential smoothing + Fourier attention | [Woo et al., 2022](https://arxiv.org/abs/2202.01381) |
| Basisformer | Learnable seasonal-trend basis | — |
| FiLM | Frequency-improved legendre memory | [Zhou et al., NeurIPS 2022](https://arxiv.org/abs/2205.08897) |
| CATS | Channel attention transformer | — |

#### MLP / Linear family

| Model | Key Idea | Paper |
|-------|----------|-------|
| DLinear | Simple decomposition + linear | [Zeng et al., AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26317) |
| NLinear | Normalised linear | [Zeng et al., AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26317) |
| RLinear | Reversible normalisation + linear | — |
| LightTS | Interval-enhanced MLP | [Zhang et al., 2022](https://arxiv.org/abs/2207.01186) |
| TSMixer | MLP-Mixer for time series | [Chen et al., 2023](https://arxiv.org/abs/2303.06053) |
| TiDE | Time-series dense encoder | [Das et al., 2023](https://arxiv.org/abs/2304.08424) |
| FreTS | Frequency-domain MLP | [Yi et al., NeurIPS 2023](https://arxiv.org/abs/2311.06184) |
| FITS | Frequency interpolation for time series | [Xu et al., ICLR 2024](https://openreview.net/forum?id=bWcnvZ3qMb) |
| SparseTSF | Sparse time series forecaster | [Han et al., ICML 2024](https://arxiv.org/abs/2405.00946) |
| HDMixer | Hierarchical dependency mixer | — |
| PatchMixer | Patch-based MLP-Mixer | — |
| MICN | Multi-scale isometric convolution network | [Wang et al., ICLR 2023](https://openreview.net/forum?id=zt53IDUR1U) |
| CycleNet | Learnable periodic cycle buffer | — |
| FilterNet | Learnable frequency filter bank | — |
| GatedMLPForecaster | Gated MLP with channel mixing | — |

#### CNN / TCN family

| Model | Key Idea | Paper |
|-------|----------|-------|
| SCINet | Sample convolution + interaction | [Liu et al., NeurIPS 2022](https://arxiv.org/abs/2106.09305) |
| TimesNet | TimesBlock: 2-D temporal variation | [Wu et al., ICLR 2023](https://openreview.net/forum?id=ju_Uqw384Oq) |
| ModernTCN | Modern temporal convolutional network | — |
| WaveNet | Dilated causal convolutions + gating | [van den Oord et al., 2016](https://arxiv.org/abs/1609.03499) |
| TCNForecaster | Vanilla TCN | — |
| MultiscaleConvForecaster | Parallel multi-scale conv branches | — |
| SincNetForecaster | SincNet learnable band-pass filters | — |
| WaveletForecaster | Learnable wavelet filter bank | — |
| TemporalConvAttentionForecaster | TCN + attention hybrid | — |

#### RNN / SSM / Hybrid family

| Model | Key Idea | Paper |
|-------|----------|-------|
| RNNForecaster | Vanilla LSTM/GRU | — |
| BiLSTMForecaster | Bidirectional LSTM | — |
| SegRNN | Segment-based RNN | [Lin et al., ICLR 2024](https://openreview.net/forum?id=jeqE7rqz2L) |
| Koopa | Koopman operator + Fourier | [Liu et al., NeurIPS 2023](https://arxiv.org/abs/2305.18803) |
| SOFTS | Scalable output-free time series | [Han et al., NeurIPS 2024](https://arxiv.org/abs/2404.04997) |
| TimeMixer | Decomposition + mixing at multiple scales | [Wang et al., ICLR 2024](https://openreview.net/forum?id=7oLshfEIC2) |
| N-BEATS | Neural basis expansion | [Oreshkin et al., ICLR 2020](https://openreview.net/forum?id=r1ecqn4YwB) |
| N-HiTS | Hierarchical interpolation | [Challu et al., AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26253) |
| DishTS | Distribution shift-aware | — |
| MambaForecaster | Selective state space model | [Gu & Dao, 2023](https://arxiv.org/abs/2312.00752) |
| iMamba | Inverted Mamba | — |
| SMamba | Spatial Mamba | — |
| HGRN2Forecaster | Hierarchical gated recurrent network | — |
| S4Forecaster | Structured state space (S4) | [Gu et al., ICLR 2022](https://openreview.net/forum?id=uYLFoz1vlAC) |
| LRUForecaster | Linear recurrent unit | — |
| MinGRUForecaster | Minimal gated recurrent unit | — |
| xLSTMForecaster | Extended LSTM | — |
| QRNNForecaster | Quasi-recurrent neural network | — |

#### Attention / Hybrid variants

| Model | Key Idea |
|-------|----------|
| LinearAttentionForecaster | Linear attention (O(N) complexity) |
| NystromForecaster | Nyström approximation attention |
| DiffTransformerForecaster | Differential attention |
| AFTForecaster | Attention-free transformer |
| MEGAForecaster | Exponential moving average + gated attention |
| FastFormerForecaster | Additive attention |
| HyenaForecaster | Long convolution operator |
| RetForecaster | Retentive network |
| RWKVForecaster | RWKV: linear-complexity RNN+Attn hybrid |
| ConformerForecaster | Convolution + transformer (speech-inspired) |
| GLAForecaster | Gated linear attention |

#### Specialised / Research models

| Model | Key Idea |
|-------|----------|
| GATForecaster | Graph attention network on channels |
| GCNForecaster | Graph convolutional network |
| KANForecaster | Kolmogorov-Arnold network |
| HarmonicForecaster | Learnable harmonic oscillators |
| EchoStateForecaster | Echo state / reservoir computing |
| TSReservoir | Temporal reservoir network |
| ImplicitNeuralForecaster | Implicit neural representation |
| PrototypicalForecaster | Prototype-based forecasting |
| FourierMixerForecaster | Fourier-domain channel mixing |
| RandomFourierForecaster | Random Fourier features |
| AdaptiveSpectralForecaster | Adaptive spectral filtering |
| HyperForecaster | Hypernetwork-generated weights |
| SpikeForecaster | Spiking neural network |
| MoEForecaster | Mixture of experts |
| LiquidNetForecaster | Liquid neural network (CfC) |
| DualDecompForecaster | Dual-branch trend/seasonal decomposition |
| NeuralBasisForecaster | Neural basis functions |
| TFT | Temporal fusion transformer | 

---

### Implemented Models — Probabilistic Forecasters

| Model | Method | Output |
|-------|--------|--------|
| GaussianForecaster | Gaussian NLL head | μ, σ per step |
| QuantileForecaster | Pinball loss | Quantile levels |
| MCDropoutForecaster | MC Dropout | Sample trajectories |
| StudentTForecaster | Student-t NLL head | μ, ν, σ per step |
| NormalizingFlowForecaster | Real-NVP conditional flow | Exact-likelihood samples |
| EnsembleForecaster | Deep ensemble | Sample trajectories |

```python
from torch_timeseries.model import GaussianForecaster

model = GaussianForecaster(seq_len=96, pred_len=24, enc_in=7)
mu, sigma = model(x)   # both (B, 24, 7)
```

---

### Implemented Models — Generation

Time series generation models produce synthetic sequences from noise or
latent codes.  All support `(seq_len, n_features)` generation calls.

| Model | Method | Paper |
|-------|--------|-------|
| TimeGAN | Adversarial training in latent space | [Yoon et al., NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html) |
| DiffusionTS | Transformer-based DDPM | [Yuan & Qiao, ICLR 2024](https://openreview.net/forum?id=4h1apFjO99) |
| NsDiff | Non-stationary diffusion | — |
| TMDM | Transformer masked diffusion model | — |
| CSDI | Conditional score-based diffusion | [Tashiro et al., NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html) |
| TimeDiff | Time series diffusion | — |

```python
from torch_timeseries.model import DiffusionTS

gen   = DiffusionTS(seq_len=96, n_features=7, n_diffusion_steps=50)
noise = torch.randn(16, 96, 7)
synth = gen.sample(noise)    # (16, 96, 7)
```

---

### Implemented Models — Irregular Time Series

Models that handle variable-length sequences with arbitrary timestamps and
possibly missing channels.

| Model | Classify | Interp. | Forecast | Extra deps |
|-------|----------|---------|----------|------------|
| [GRU-D (2018)](https://www.nature.com/articles/s41598-018-24271-9) | ✅ | ✅ | ✅ | — |
| [mTAN (2021)](https://openreview.net/forum?id=4c0J6lwQ4_) | ✅ | ✅ | ✅ | — |
| [LatentODE (2019)](https://proceedings.neurips.cc/paper/2019/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html) | ✅ | ✅ | ✅ | torchdiffeq |
| [NeuralCDE (2020)](https://proceedings.neurips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html) | ✅ | | | torchcde |
| [Raindrop (2022)](https://openreview.net/forum?id=Kwm8I7dU-l5) | ✅ | | | torch_geometric |

```python
from torch_timeseries.model.irregular import GRUD

model = GRUD(input_size=35, hidden_size=64, n_classes=2)
# batch.values: (B, T, 35)  batch.mask: (B, T, 35)  batch.deltas: (B, T, 35)
logits = model(batch.values, batch.mask, batch.deltas)   # (B, 2)
```



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
