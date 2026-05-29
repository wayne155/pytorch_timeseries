# Time Series Generation — Design Spec

**Date:** 2026-05-29

## Goal

Add a `GenerationExp` experiment type that trains generative models on fixed-length time series windows and evaluates them with standard synthesis-quality metrics (FID-TS, MMD, discriminative score, predictive score). Phase 1 includes three architectures: TimeGAN, LSTMVAE, and DDPM-TS.

## Architecture Overview

No new DataModule is needed. `ForecastDataModule` already produces `TSBatch` with `x: (B, T, F)` windows, which is exactly the training target for generative models. The new layer is a thin `GenerationExp` base class, three model files, and a generation metrics module.

```
ForecastDataModule (existing)
     ↓  TSBatch.x  (B, T, F)
GenerationExp.train_epoch()
     ↓  model.loss(batch)  → scalar
   [train loop]
     ↓  model.generate(n)  → Tensor(n, T, F)
GenerationExp.evaluate()
     ↓  generation metrics
  Dict[str, float]: fid, mmd, discriminative_score, predictive_score
```

## File Structure

| File | Role |
|------|------|
| `torch_timeseries/experiments/generation.py` | `GenerationExp` base class + `GenerationSettings` |
| `torch_timeseries/model/generation/__init__.py` | Package exports |
| `torch_timeseries/model/generation/timegan.py` | TimeGAN model |
| `torch_timeseries/model/generation/lstmvae.py` | LSTMVAE model |
| `torch_timeseries/model/generation/ddpm_ts.py` | DDPM-TS model |
| `torch_timeseries/metrics/generation.py` | FID-TS, MMD, discriminative score, predictive score |
| `torch_timeseries/experiments/TimeGAN.py` | `TimeGANGeneration` combo |
| `torch_timeseries/experiments/LSTMVAE.py` | `LSTMVAEGeneration` combo |
| `torch_timeseries/experiments/DDPMTS.py` | `DDPMTSGeneration` combo |
| `torch_timeseries/experiments/registry.py` | Add `"Generation"` to `TASK_SUFFIXES` |
| `torch_timeseries/experiments/__init__.py` | Import three combo classes |
| `tests/experiments/test_generation.py` | Smoke tests |
| `tests/metrics/test_generation_metrics.py` | Unit tests for metrics |

## `GenerationSettings` Dataclass

```python
@dataclass
class GenerationSettings:
    seq_len: int = 96
    num_generated: int = 1000   # samples to draw at evaluation time
    batch_size: int = 32
    train_epochs: int = 10
    learning_rate: float = 1e-3
```

`num_features` is inferred from the dataset at `_init_data_loader` time (not a setting).

## `GenerationExp` Base Class

```python
class GenerationExp(BaseRelevant, BaseIrrelevant, GenerationSettings):
    """Base experiment for unconditional time series generation."""

    # --- abstract -------------------------------------------------------
    def _init_model(self) -> None: ...         # sets self.model
    def generate(self, n: int) -> Tensor: ...  # returns (n, T, F)

    # --- concrete -------------------------------------------------------
    def _init_data_loader(self) -> None:
        # builds ForecastDataModule with pred_len=1 (minimum valid)
        # batch.y is unused; only batch.x (B, T, F) is passed to model.loss()
        # sets self.num_features, self.dm

    def _train_epoch(self, epoch: int) -> float:
        # iterates dm.train_loader, calls model.loss(batch.x), returns mean loss

    def run(self, seed: int = 42) -> Dict[str, float]:
        # 1. reproduce(seed)
        # 2. _init_data_loader(); _init_model(); _init_optimizer()
        # 3. train loop (train_epochs)
        # 4. generate(num_generated) → fake
        # 5. collect real test samples → real
        # 6. compute and return metrics
```

The training loop contract is: `model.loss(batch_x: Tensor) → scalar Tensor`. Each model owns its own loss computation (adversarial/ELBO/score matching). The base class never calls `criterion()` directly.

## Model Interfaces

All three models expose:

```python
class TimeSeriesGenerativeModel(nn.Module):
    def loss(self, x: Tensor) -> Tensor:
        """x: (B, T, F). Returns scalar loss."""

    def generate(self, n: int, device: torch.device) -> Tensor:
        """Returns (n, T, F) samples."""
```

### TimeGAN (`model/generation/timegan.py`)

Five networks: `embedder`, `recovery`, `generator`, `supervisor`, `discriminator`. Training has three phases: (1) autoencoder warmup, (2) supervised step, (3) joint adversarial. The `loss()` method advances whichever phase is current (tracked internally with a step counter).

Parameters:
```python
@dataclass
class TimeGANParameters:
    hidden_size: int = 24
    num_layers: int = 3
    noise_dim: int = 24
    gamma: float = 1.0   # discriminator loss weight
```

### LSTMVAE (`model/generation/lstmvae.py`)

Bidirectional LSTM encoder → reparameterize → LSTM decoder. `loss(x)` = MSE reconstruction + β×KL. `generate(n)` samples latent z ~ N(0,I) and decodes.

Parameters:
```python
@dataclass
class LSTMVAEParameters:
    hidden_size: int = 64
    latent_dim: int = 32
    num_layers: int = 2
    beta: float = 1.0   # KL weight
```

### DDPM-TS (`model/generation/ddpm_ts.py`)

1D UNet with temporal self-attention conditioned on diffusion timestep embedding. Standard DDPM with cosine noise schedule. `loss(x)` = MSE(predicted_noise, actual_noise). `generate(n)` runs full reverse diffusion from z ~ N(0,I).

Parameters:
```python
@dataclass
class DDPMTSParameters:
    unet_dim: int = 64
    unet_dim_mults: tuple = (1, 2, 4)
    T_diffusion: int = 1000
    beta_schedule: str = "cosine"  # "linear" | "cosine"
```

## Metrics (`torch_timeseries/metrics/generation.py`)

All functions accept `real: Tensor (N, T, F)` and `fake: Tensor (M, T, F)` and return `float`.

### `fid_ts(real, fake, feature_extractor=None)`

Train a simple 2-layer LSTM feature extractor on `real` (next-step prediction, 10 epochs). Extract penultimate-layer features from real and fake. Compute FID: `‖μ_r - μ_f‖² + Tr(Σ_r + Σ_f - 2(Σ_rΣ_f)^½)`. Returns lower-is-better float.

If a pre-trained `feature_extractor` is provided (from a previous call), reuse it to avoid re-training.

### `mmd(real, fake, kernel='rbf', gamma=1.0)`

Maximum mean discrepancy using RBF kernel on flattened (N, T×F) sequences. `MMD² = E[k(r,r')] - 2E[k(r,f)] + E[k(f,f')]`. Returns float (lower = more similar).

### `discriminative_score(real, fake)`

Train a binary 2-layer LSTM + sigmoid classifier (80/20 train/test split on the combined set) for 10 epochs. Metric = |test_accuracy − 0.5|. Perfect generation → 0.0 (classifier at chance); easily-distinguished → 0.5.

### `predictive_score(real, fake)`

TSTR: train a 2-layer LSTM on fake to predict the next step (t+1 given 1:t). Evaluate MAE on real. Returns float (lower = more useful fake data).

All four helper models (feature extractor, discriminator, TSTR predictor) use CPU by default; moved to CUDA if available.

## Experiment Combo Classes

```python
# experiments/TimeGAN.py
@dataclass
class TimeGANGeneration(GenerationExp, TimeGANParameters):
    model_type: str = "TimeGAN"
    def _init_model(self): self.model = TimeGAN(...)

# experiments/LSTMVAE.py
@dataclass
class LSTMVAEGeneration(GenerationExp, LSTMVAEParameters):
    model_type: str = "LSTMVAE"
    def _init_model(self): self.model = LSTMVAE(...)

# experiments/DDPMTS.py
@dataclass
class DDPMTSGeneration(GenerationExp, DDPMTSParameters):
    model_type: str = "DDPMTS"
    def _init_model(self): self.model = DDPMTimeSeries(...)
```

Registry entry examples: `("TimeGAN", "Generation")`, `("LSTMVAE", "Generation")`, `("DDPMTS", "Generation")`.

## Registry Change

In `torch_timeseries/experiments/registry.py`:

```python
TASK_SUFFIXES = (
    "Forecast", "Imputation", "UEAClassification",
    "AnomalyDetection", "IrregularClassification",
    "Generation",   # ← new
)
```

## Error Handling

- If `num_generated` > training set size, log a warning but proceed (metrics will still be valid).
- Metric helper models (discriminator, TSTR) should catch convergence failures and return `float('nan')` rather than crashing the experiment.
- DDPM reverse diffusion: clip samples to `[μ - 3σ, μ + 3σ]` of training data after generation.

## Testing

**`tests/experiments/test_generation.py`**:
- `test_timegan_smoke` — 2 epochs, tiny dataset (5 samples, T=8, F=3), check `run()` returns dict with expected keys
- `test_lstmvae_smoke` — same
- `test_ddpmts_smoke` — same with T_diffusion=10 (fast)
- `test_generation_registry` — `get_experiment_class("TimeGAN", "Generation")` returns correct class
- `test_generate_shape` — `model.generate(10)` returns shape `(10, T, F)`

**`tests/metrics/test_generation_metrics.py`**:
- `test_fid_identical` — real == fake → FID ≈ 0
- `test_mmd_identical` — real == fake → MMD ≈ 0
- `test_discriminative_identical` — indistinguishable → score near 0
- `test_metrics_shapes` — all four functions accept (N, T, F) and return float
