# Time Series Generation — Design Spec

**Date:** 2026-06-16  
**Status:** approved  
**Scope:** Add `Generation` as a sixth task to `torch-timeseries`, implementing five generative models and a suite of four evaluation metrics.

---

## 1. Overview

The library currently covers five tasks (Forecast, Imputation, UEAClassification, AnomalyDetection, IrregularClassification). This spec adds **Generation** as a first-class sixth task following the same `{Model}{Task}` naming convention so all five new models integrate with the existing `Experiment` builder, CLI, result backends, and leaderboard.

### Five models

| Model | Paper | Type |
|---|---|---|
| TimeGAN | Yoon et al., 2019 | GAN (multi-phase) |
| CSDI | Tashiro et al., 2021 | Score-based diffusion |
| Diffusion-TS | Yuan & Qiao, 2024 | DDPM + Transformer decoder |
| TimeDiff | Shen & Kwok, 2023 | Self-guidance diffusion |
| NS-Diffusion | Ye (this project) | Non-stationary adaptive diffusion |

### Four evaluation metrics

Discriminative score · Predictive score · Context-FID · Correlational score.

---

## 2. New components

```
torch_timeseries/
  dataloader/v2/
    generation.py           # GenerationDataModule, GenerationWindowConfig
  experiments/
    generation.py           # GenerationExp base class
    TimeGAN.py              # TimeGANGeneration
    CSDI.py                 # CSDIGeneration
    DiffusionTS.py          # DiffusionTSGeneration
    TimeDiff.py             # TimeDiffGeneration
    NSDiffusion.py          # NSDiffusionGeneration
  model/
    TimeGAN.py
    CSDI.py
    DiffusionTS.py
    TimeDiff.py
    NSDiffusion.py
  metrics/
    generation.py           # discriminative_score, predictive_score,
                            #   context_fid, correlational_score

tests/
  model/test_generation_models.py
  experiments/test_generation_experiments.py
  metrics/test_generation_metrics.py

leaderboard/reproduce/generation/
  timegan.py
  csdi.py
  diffusion_ts.py
  timediff.py
  ns_diffusion.py
```

### Registry change (one line)

`torch_timeseries/experiments/registry.py`:
```python
TASK_SUFFIXES = ("Forecast", "Imputation", "UEAClassification", "AnomalyDetection",
                 "IrregularClassification", "Generation")
```

---

## 3. Data pipeline

### `GenerationWindowConfig`

Simpler than `WindowConfig` — generation doesn't need `horizon` or `steps`:

```python
@dataclass
class GenerationWindowConfig:
    seq_len: int = 96          # window length fed to the model
    stride: int = 1            # sliding stride during training
    fast_eval: bool = False    # non-overlapping windows at eval time
```

### `GenerationDataModule`

```python
class GenerationDataModule:
    def __init__(
        self,
        dataset: TimeSeriesDataset,
        scaler: Scaler,
        window: GenerationWindowConfig,
        split: Optional[SplitConfig] = None,   # None → dataset default
        loader: LoaderConfig = LoaderConfig(),
    ): ...

    @property
    def train_loader(self) -> DataLoader: ...
    @property
    def test_loader(self) -> DataLoader: ...
```

Each batch is a **`TSBatch`** (reused unchanged) with:
- `x`: `(B, seq_len, C)` — scaled real sequence (the thing being generated/learned)
- `y`: `(B,)` or `None` — optional class label (for conditional models)
- `x_time`: time features (optional)
- `x_index`: integer positions

The `y_*` forecasting fields are always `None` in generation batches.

---

## 4. `GenerationExp` base class

```python
@dataclass
class GenerationExp(BaseRelevant, BaseIrrelevant):
    seq_len: int = 96
    train_ratio: Optional[float] = None
    test_ratio: Optional[float] = None
    # generative training defaults
    epochs: int = 200
    patience: int = 30          # patience on plateau of train loss
    lradj: str = "cosine"
    eval_n_samples: int = 1000  # samples generated for metric evaluation
```

### Training loop

```
for epoch in epochs:
    for batch in train_loader:
        loss = _process_train_batch(batch)   ← subclass implements
        loss.backward(); optimizer.step()
    check early_stop on rolling-window train loss plateau
```

Early stopping monitors the rolling mean of training loss (no val split needed). `patience` is the number of epochs without improvement.

### Evaluation

After training:
```python
real = collect_test_sequences(test_loader)  # (N, seq_len, C)
fake = self.generate(len(real))             # (N, seq_len, C)
metrics = {
    "discriminative_score": discriminative_score(real, fake),
    "predictive_score":     predictive_score(real, fake),
    "context_fid":          context_fid(real, fake),
    "correlational_score":  correlational_score(real, fake),
}
```

### Subclass contract

```python
def _init_model(self) -> None:
    """Instantiate self.model and move to self.device."""
    raise NotImplementedError()

def _process_train_batch(self, batch: TSBatch) -> Tensor:
    """Return a scalar loss tensor for one batch."""
    raise NotImplementedError()

def generate(self, n_samples: int, condition=None) -> Tensor:
    """Return (n_samples, seq_len, num_features) on CPU."""
    raise NotImplementedError()
```

TimeGAN overrides `run()` entirely due to its multi-phase training.

---

## 5. Model architectures

### 5.1 TimeGAN

File: `model/TimeGAN.py`

Five sub-networks, all GRU-based:
- **Embedder** `e(X) → H` — 3-layer GRU, input dim → hidden_dim
- **Recovery** `r(H) → X̂` — 3-layer GRU, hidden_dim → input dim
- **Generator** `g(Z) → Ê` — 3-layer GRU, input dim → hidden_dim
- **Supervisor** `s(H) → Ĥ` — 2-layer GRU, hidden_dim → hidden_dim
- **Discriminator** `d(H) → ŷ` — 3-layer GRU + linear → scalar

**Default hyperparams:** `hidden_dim=24, n_layers=3, gamma=1.0`

**Training phases** (all run inside `TimeGANGeneration.run()`):

| Phase | Networks trained | Loss |
|---|---|---|
| 1. Autoencoder | Embedder + Recovery | MSE reconstruction |
| 2. Supervised | Generator + Supervisor | MSE supervised (real latent) |
| 3. Joint | All five | `L_S + L_U + γ · L_G` |

Phase epochs: `epochs_ae`, `epochs_sup`, `epochs_joint` (each defaults to `epochs`).

`generate(n_samples)`: sample Z ~ N(0,1), pass through Generator → Recovery → (n, seq_len, C).

### 5.2 CSDI

File: `model/CSDI.py`

Score-based diffusion for time series.

**Noise schedule:** linear β from 1e-4 to 0.02, T=100 steps.

**Denoising network:** stack of `ResidualBlock` modules, each containing:
- 1D temporal attention over time axis
- 1D feature attention over channel axis
- Sinusoidal timestep embedding injected at each block

**Conditioning:** mask-based. For unconditional generation, `x_cond = 0`, `mask = 0`. For conditional, observed channels are provided as conditioning.

**Hyperparams:** `d_model=64, n_heads=8, n_layers=4, T=100`

**`generate(n)`:** reverse diffusion from `x_T ~ N(0,I)` using DDPM ancestral sampling.

### 5.3 Diffusion-TS

File: `model/DiffusionTS.py`

DDPM with explicit trend + seasonal decomposition at each denoising step.

**Denoising network:**
- Input: `x_t ∈ (B, seq_len, C)` + timestep t
- **Trend branch:** linear layer over time dim → `trend ∈ (B, seq_len, C)`
- **Seasonal branch:** Transformer decoder (d_model, n_heads, n_layers) → FFT-based harmonics → `seasonal ∈ (B, seq_len, C)`
- Output: `ε_θ(x_t, t) = trend + seasonal`

**Noise schedule:** cosine schedule (Nichol & Dhariwal 2021), T=1000.

**Hyperparams:** `d_model=128, n_heads=4, n_layers=4, n_harmonics=8, T=1000`

**`generate(n)`:** standard DDPM reverse with DDIM accelerated sampling (50 steps).

### 5.4 TimeDiff

File: `model/TimeDiff.py`

Self-guidance diffusion via **future mixup** and **target interpolation**.

**Key additions over standard DDPM:**
- **Future mixup:** during training, `x̃ = α · x + (1-α) · ε` where α ~ Beta(0.5, 0.5); the model sees a noisy version of the target as auxiliary conditioning
- **Target interpolation:** at inference, mix diffusion samples with a simple trend estimate as self-guidance

**Denoising network:** Transformer with rotary positional encoding, `d_model=128, n_heads=4, n_layers=4`.

**Noise schedule:** linear, T=500.

**Hyperparams:** `d_model=128, n_heads=4, n_layers=4, T=500, mix_ratio=0.5`

**`generate(n)`:** DDPM reverse with self-guidance correction.

### 5.5 NS-Diffusion (Ye)

File: `model/NSDiffusion.py`

Non-stationary adaptive diffusion — handles non-stationary time series (drifting mean/variance).

**Core idea:**
- Standard DDPM assumes a stationary prior; real-world TS are non-stationary.
- The forward process adapts β_t per-segment based on local statistics (mean μ_t, std σ_t of `x`).
- The denoising network uses **Reversible Instance Normalization (RevIN)** to remove non-stationarity before denoising, then re-injects it.

**Components:**
- `NSDiffusionSchedule`: learns `β_t(x)` conditioned on `(μ, σ)` of the input sequence; parameterized as a small MLP mapping `[μ, σ, t] → β_t`
- `NSDenoiser`: Transformer (d_model, n_heads, n_layers) wrapped with RevIN; receives `x_t`, `t`, and `(μ_0, σ_0)` of the original sequence
- `NSDiffusion`: coordinates forward process, reverse process, and adaptive schedule

**Forward process:**
```
x_t = sqrt(ᾱ_t(x_0)) · x_0 + sqrt(1 - ᾱ_t(x_0)) · ε
where ᾱ_t depends on local statistics of x_0
```

**Training loss:** simplified DDPM objective with KL regularization on the adaptive schedule.

**Hyperparams:** `d_model=128, n_heads=4, n_layers=4, T=500`

> **Note for author:** The skeleton provides the RevIN wrapper, Transformer backbone, and adaptive schedule MLP. Fill in `_adaptive_beta()` and any architectural details from the paper.

---

## 6. Evaluation metrics (`metrics/generation.py`)

All four metrics take `real: Tensor (N, seq_len, C)` and `fake: Tensor (N, seq_len, C)` as inputs. All are computed on CPU with NumPy/PyTorch, no external ML library required.

### 6.1 Discriminative score

Train a 2-layer LSTM binary classifier (hidden=64) to distinguish real from fake.  
Run 3 random train/test splits (70/30).  
`discriminative_score = mean(|accuracy - 0.5|)` over splits.  
Lower = better (0 → perfectly indistinguishable).

### 6.2 Predictive score

Train a 2-layer GRU seq2seq (hidden=64) on **fake** data: predict step `t+1` from steps `0..t`.  
Evaluate MAE on **real** test sequences.  
Lower = better (good synthetic data preserves temporal dynamics).

### 6.3 Context-FID

Encode both sets with a shared **2-layer GRU** encoder (hidden=64), using the final hidden state as embedding → `(N, 64)`.  
Compute Fréchet distance: `||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2√(Σ_r Σ_f))`.  
Uses `scipy.linalg.sqrtm`. Lower = better.

### 6.4 Correlational score

For each sequence compute the per-feature autocorrelation at lags 1..K (default K=20).  
`correlational_score = MSE(autocorr(real), autocorr(fake))` averaged over features.  
Lower = better.

---

## 7. Integration

### CLI

```bash
pytexp TimeGAN Generation ETTh1 run 1
pytexp DiffusionTS Generation ETTh1 --seq_len 96 run 1
```

### Experiment builder

```python
from torch_timeseries import Experiment

Experiment("TimeGAN", "Generation", "ETTh1", seq_len=96, epochs=200) \
    .with_local("./results") \
    .run(seeds=[1, 2, 3])
```

### `experiments/__init__.py`

All five new `*Generation` classes are imported here so the registry picks them up.

### `dataloader/v2/__init__.py`

Export `GenerationDataModule`, `GenerationWindowConfig`.

---

## 8. Testing strategy

### `tests/model/test_generation_models.py`

For each model (TimeGAN, CSDI, DiffusionTS, TimeDiff, NSDiffusion):
- `forward()` with `(B=2, seq_len=16, C=3)` produces expected output shape
- `generate(n=4)` produces `(4, seq_len, C)` on CPU

### `tests/experiments/test_generation_experiments.py`

For each `*Generation` experiment class:
- `run(seed=1)` on a toy in-memory dataset (50 samples, seq_len=8, C=2) returns a dict with keys `discriminative_score, predictive_score, context_fid, correlational_score`
- Metrics are finite floats

### `tests/metrics/test_generation_metrics.py`

- Discriminative score = ~0 when real=fake (trivial case)
- Predictive score computes without error on random tensors
- Context-FID = 0 when real=fake
- Correlational score = 0 when real=fake

---

## 9. Dependencies

No new hard dependencies. `scipy` is a transitive dependency of `scikit-learn` and `sktime`. No specialized diffusion library needed — all models are pure PyTorch.

---

## 10. Out of scope

- Class-conditional generation beyond a simple label embedding
- ONNX/torchscript export
- Distributed/multi-GPU training for generative models
- Automatic hyperparameter search
