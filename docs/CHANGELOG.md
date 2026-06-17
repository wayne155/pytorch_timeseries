# Changelog

## 0.2.5

- Add temporal encoding classes to `torch_timeseries.nn`: `Time2Vec`, `LearnableFourierFeatures`, `RotaryEmbedding` (RoPE), `SinusoidalEmbedding`.
- Add Sphinx model reference — all 20 models documented with paper citations and Args.
- Add Sphinx nn reference — temporal encoding and embedding classes.
- Restructure README: new TOC covering all 6 time-series tasks; full code examples and figures for Imputation, Anomaly Detection, and Classification.
- Fix hard-coded `.cuda()` → `.to(device)` in `AutoCorrelation.py`.
- Remove duplicate class definitions in `Autoformer_EncDec.py` (aliases to canonical `kernels`/`decomp` implementations).
- Fix import shadowing in `nn/__init__.py`; remove dead code across `Crossformer.py`, `DLinear.py`, `embedding.py`, `MultiWaveletCorrelation.py`.

## 0.2.4

- Add `NsDiff` (Non-Stationary DDPM) and `TMDM` generation models.
- Add `Sine` and `Stocks` benchmark generation datasets.
- Add `NsDiffGeneration` and `TMDMGeneration` experiment wrappers.
- Rename `NSDiffusion` → `NsDiff` throughout.

## 0.2.3

- Add probabilistic forecasting via `ProbForecastExp`.
- Add probabilistic metrics in `torch_timeseries.metrics`: CRPS, CRPSSum, QICE, PICP, ProbMSE, ProbMAE, and ProbRMSE.
- Document `WindowConfig.fast_val` and `fast_test` for non-overlapping validation/test windows.
- Refactor forecast batch handling to structured `TSBatch` inputs.

## 0.2.2

- Add `build_dataset(csv=..., freq=...)` for local CSV datasets.
- Infer dataset `num_features` and `length` from loaded data.
- Remove the runtime `torchvision` dependency.
- Add README guidance for custom datasets and compact curated leaderboard entries.

## 0.2.1

- Add per-task leaderboard reproduce scripts for anomaly detection, imputation, long-term forecast, short-term forecast, and UEA classification across DLinear, NLinear, PatchTST, iTransformer, TimesNet, Autoformer, and FEDformer.
- Add NLinear model and experiment wrappers.
- Add TimesNet imputation and anomaly detection experiment wrappers.
- Add README examples for researcher workflows using custom dataloaders, custom training loops, and default experiment registration.
- Add default ForecastExp batch handling for simple models that map `batch_x` to forecast predictions.
- Normalize anomaly metric names for leaderboard rendering.
