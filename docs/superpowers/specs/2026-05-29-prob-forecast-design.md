# Probabilistic Time Series Forecasting — Design Spec

**Date:** 2026-05-29

## Goal

Add a `ProbForecastExp` experiment type for uncertainty-aware time series forecasting. Models output a distribution over future values by generating N samples; the base class evaluates with sample-based calibration metrics (CRPS, WIS, empirical coverage, NLL). Phase 1 includes one model: CSDI (Conditional Score-based Diffusion for Imputation, adapted for forecasting).

## Architecture Overview

`ProbForecastExp` is a parallel base class to `ForecastExp` — same DataModule, same training loop template. The key difference is the inference interface: instead of a single point prediction, models return `(B, S, H, F)` sample tensors at evaluation time. The base class never inspects the model's internal loss; models own that entirely.

```
ForecastDataModule (existing)
     ↓  TSBatch: x(B,T,F), y(B,H,F)
ProbForecastExp.train_epoch()
     ↓  model.loss(batch)  → scalar
   [train loop]
     ↓  model.forward_samples(batch, n_samples)  → (B, S, H, F)
ProbForecastExp.evaluate()
     ↓  probabilistic metrics
  Dict[str, float]: crps, wis, coverage_90, mae, mse
```

## File Structure

| File | Role |
|------|------|
| `torch_timeseries/experiments/prob_forecast.py` | `ProbForecastExp` base class + `ProbForecastSettings` |
| `torch_timeseries/model/prob_forecast/__init__.py` | Package exports |
| `torch_timeseries/model/prob_forecast/csdi.py` | CSDI model |
| `torch_timeseries/metrics/probabilistic.py` | CRPS, WIS, coverage, NLL |
| `torch_timeseries/experiments/CSDI.py` | `CSDIProbForecast` combo |
| `torch_timeseries/experiments/registry.py` | Add `"ProbForecast"` to `TASK_SUFFIXES` |
| `torch_timeseries/experiments/__init__.py` | Import `CSDIProbForecast` |
| `tests/experiments/test_prob_forecast.py` | Smoke tests |
| `tests/metrics/test_probabilistic_metrics.py` | Unit tests for metrics |

## `ProbForecastSettings` Dataclass

```python
@dataclass
class ProbForecastSettings:
    seq_len: int = 96
    pred_len: int = 96
    n_samples: int = 100     # MC samples drawn at evaluation time
    batch_size: int = 32
    train_epochs: int = 10
    learning_rate: float = 1e-3
    eval_quantile_levels: tuple = (0.1, 0.5, 0.9)  # used for WIS + coverage
```

`num_features` is inferred from the dataset at `_init_data_loader` time.

## `ProbForecastExp` Base Class

```python
class ProbForecastExp(BaseRelevant, BaseIrrelevant, ProbForecastSettings):
    """Base experiment for sample-based probabilistic time series forecasting."""

    # --- abstract -------------------------------------------------------
    def _init_model(self) -> None: ...
    def forward_samples(self, batch: TSBatch, n_samples: int) -> Tensor: ...
    #   ^ returns (B, S, H, F) — S independent samples of the future

    # --- concrete -------------------------------------------------------
    def _init_data_loader(self) -> None:
        # builds ForecastDataModule(seq_len, pred_len, ...)
        # sets self.num_features, self.dm

    def _train_epoch(self, epoch: int) -> float:
        # iterates dm.train_loader, calls model.loss(batch), returns mean loss

    def run(self, seed: int = 42) -> Dict[str, float]:
        # 1. reproduce(seed)
        # 2. _init_data_loader(); _init_model(); _init_optimizer()
        # 3. train loop (train_epochs)
        # 4. evaluate on dm.test_loader:
        #    - calls forward_samples(batch, n_samples) → samples (B, S, H, F)
        #    - accumulates all samples and targets
        # 5. compute and return metrics:
        #    wis() receives self.eval_quantile_levels
        #    empirical_coverage() uses max(eval_quantile_levels) as level
```

The training contract is: `model.loss(batch: TSBatch) → scalar Tensor`. The base class only calls `loss()` during training and `forward_samples()` during evaluation.

## CSDI Model (`model/prob_forecast/csdi.py`)

CSDI (Tashiro et al. 2021) is a score-based diffusion model that conditions on observed values to impute/forecast masked positions. We adapt it for forecasting by treating the future horizon as the "missing" region.

### Architecture

```
CSDIForecast
├── noise_schedule: linear beta_1..beta_T (default T=50 for fast training)
├── side_info_encoder: Linear(F, d_model)  [encodes past context into conditioning]
└── denoiser: CSDIDenoiser
    ├── temporal_transformer: TransformerEncoder (L=4, d_model=64, nhead=8)
    │   attends over time dimension of noisy future
    └── cross_attn_layers: L × CrossAttention(d_model)
        cross-attends noisy future ← past context
```

Forward (training, score matching):
1. Given `x_cond (B, T, F)` and `x_target (B, H, F)`
2. Sample diffusion step `k ~ Uniform(0, T)`
3. Add noise: `x_noisy = sqrt(alpha_bar_k) * x_target + sqrt(1 - alpha_bar_k) * eps`
4. Predict `eps_pred = denoiser(x_noisy, x_cond, k)`
5. Loss = `MSE(eps_pred, eps)`

Forward (inference, sampling):
1. Start from `x_T ~ N(0, I)` shaped `(B, H, F)`
2. For `k = T..1`: compute `x_{k-1}` via DDPM update conditioned on `x_cond`
3. Repeat `n_samples` times

```python
class CSDIForecast(nn.Module):
    def loss(self, batch: TSBatch) -> Tensor:
        """Returns scalar score-matching loss."""

    def forward_samples(
        self,
        batch: TSBatch,
        n_samples: int,
        device: torch.device,
    ) -> Tensor:
        """Returns (B, n_samples, H, F) sample forecasts."""
```

Parameters:
```python
@dataclass
class CSDIParameters:
    d_model: int = 64
    nhead: int = 8
    num_layers: int = 4
    T_diffusion: int = 50    # diffusion steps (50 for training speed)
    beta_start: float = 1e-4
    beta_end: float = 0.02
```

## Metrics (`torch_timeseries/metrics/probabilistic.py`)

All functions accept `samples: Tensor (B, S, H, F)` and `target: Tensor (B, H, F)`.

### `crps_from_samples(samples, target) → float`

Energy form of CRPS:
```
CRPS(F, y) = E_X|X - y| - 0.5 * E_{X,X'}|X - X'|
```
where X, X' are independent draws from the forecast distribution. Averaged over B×H×F elements. Lower is better.

Implementation: vectorized using pairwise differences. For B=32, S=100, H=96, F=7 this is feasible without chunking on GPU.

### `wis(samples, target, levels=(0.1, 0.5, 0.9)) → float`

For each level α, compute the (α/2, 1-α/2) prediction interval from sample quantiles. WIS = sum of interval scores weighted by α/2:

```
IS_α(l, u, y) = (u - l) + (2/α) * max(l-y, 0) + (2/α) * max(y-u, 0)
WIS = (1/(K+0.5)) * [0.5 * |median - y| + sum_k (α_k/2) * IS_αk]
```

Returns the mean WIS over B×H×F elements.

### `empirical_coverage(samples, target, level=0.9) → float`

Compute the (level/2, 1-level/2) quantile bounds per element from `samples`. Return the fraction of `target` elements that fall within bounds. Ideal value = level.

### `nll_from_samples(samples, target, bandwidth=None) → float`

Kernel density estimate NLL. For each element (b, h, f), fit a 1D KDE to the S samples using Silverman bandwidth (or provided), evaluate log-density at `target[b, h, f]`. Return mean negative log-density. Lower is better (more probability mass on the true value).

### `mae_from_samples(samples, target) → float`

`|median(samples) - target|.mean()`. Included in `run()` output alongside probabilistic metrics.

### `mse_from_samples(samples, target) → float`

`(mean(samples) - target)^2.mean()`.

## Experiment Combo Class

```python
# experiments/CSDI.py
@dataclass
class CSDIProbForecast(ProbForecastExp, CSDIParameters):
    model_type: str = "CSDI"

    def _init_model(self):
        self.model = CSDIForecast(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            num_features=self.num_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            T_diffusion=self.T_diffusion,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
        ).to(self.device)

    def forward_samples(self, batch: TSBatch, n_samples: int) -> Tensor:
        return self.model.forward_samples(batch, n_samples, self.device)
```

Registry entry: `("CSDI", "ProbForecast")`.

## Registry Change

In `torch_timeseries/experiments/registry.py`:

```python
TASK_SUFFIXES = (
    "Forecast", "Imputation", "UEAClassification",
    "AnomalyDetection", "IrregularClassification",
    "Generation",      # from generation spec
    "ProbForecast",    # ← new
)
```

## Error Handling

- `n_samples=1`: WIS and coverage are undefined for a single sample. Log a warning and return `float('nan')` for those metrics.
- CSDI reverse diffusion: clip output to `[μ_train - 5σ_train, μ_train + 5σ_train]` to avoid extreme values from noisy early-training denoiser.
- KDE NLL: if all S samples are identical (collapsed posterior), return `float('nan')` rather than divide-by-zero.

## Testing

**`tests/experiments/test_prob_forecast.py`**:
- `test_csdi_smoke` — 2 epochs, tiny dataset (5 samples, T=8, pred_len=4, F=3), T_diffusion=5, n_samples=3; check `run()` returns dict with `crps`, `wis`, `coverage_90`, `mae`, `mse`
- `test_prob_forecast_registry` — `get_experiment_class("CSDI", "ProbForecast")` returns `CSDIProbForecast`
- `test_forward_samples_shape` — `forward_samples(batch, n_samples=5)` returns `(B, 5, H, F)`
- `test_forward_samples_gradient_disabled` — no grad on samples at eval time

**`tests/metrics/test_probabilistic_metrics.py`**:
- `test_crps_perfect` — samples concentrated at target → CRPS ≈ 0
- `test_coverage_calibrated` — samples from N(target, 0.01) → coverage ≈ 1.0
- `test_wis_perfect` — tight interval around target → small WIS
- `test_metrics_shapes` — all functions accept (B, S, H, F) / (B, H, F) and return float
- `test_nll_known_distribution` — samples from N(0,1), target=0 → NLL ≈ 0.919 (= -log(N(0,1)(0)))
