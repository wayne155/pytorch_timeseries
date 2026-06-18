# Changelog

## 0.2.9

- Add `TimeMixer` — Past Decomposable Mixing (PDM) + Future Multipredictor Mixing (FMM). Multi-scale seasonal/trend mixing with bottom-up seasonal and top-down trend cascades. Supports Forecast, Imputation, Anomaly Detection, and Classification.
- Add `TimeMixerForecast`, `TimeMixerImputation`, `TimeMixerAnomalyDetection`, `TimeMixerUEAClassification` experiment wrappers.
- Add leaderboard reproduce script for TimeMixer long-term forecast.

## 0.2.8

- Add `SegRNN` — Segment Recurrent Neural Network (Lin et al., ICLR 2024). Divides the look-back window into segments, encodes with a GRU, and iteratively decodes via the IMO strategy. Supports Forecast, Imputation, Anomaly Detection, and Classification.
- Add `SegRNNForecast`, `SegRNNImputation`, `SegRNNAnomalyDetection`, `SegRNNUEAClassification` experiment wrappers.
- Add leaderboard reproduce script for SegRNN long-term forecast.
- Add leaderboard reproduce scripts for GRUD, mTAN, LatentODE on IrregularClassification; GRUD, mTAN on IrregularInterpolation and IrregularForecast.
- Expose `irregular_models` list from `torch_timeseries.model`.
- Update model reference docs: 26 models, SegRNN card, irregular task coverage table.
- Fix CLI help text to list all 26 models and all 9 tasks including irregular variants.

## 0.2.7

- Add `mTAN` (Multi-Time Attention Network) — no external deps. Supports classification and seq2seq (interpolation/forecast) via `t_query`.
- Add `LatentODE` — variational Latent ODE with GRU encoder. All 3 irregular tasks. Requires `torchdiffeq`.
- Add `NeuralCDE` — Neural Controlled Differential Equation. Classification only. Requires `torchcde`.
- Add `Raindrop` — graph-guided sensor attention network. Classification only. Requires `torch_geometric`.
- Add `mTANIrregularClassification`, `mTANIrregularInterpolation`, `mTANIrregularForecast` experiment classes.
- Add `LatentODEIrregularClassification/Interpolation/Forecast`, `NeuralCDEIrregularClassification`, `RaindropIrregularClassification`.
- New install extra: ``pip install torch-timeseries[irregular]`` installs ``torchdiffeq``, ``torchcde``, ``torch_geometric``.

## 0.2.6

- Add `IrregularInterpolationDataModule` + `IrregularInterpolationConfig`: deterministic per-sample query holdout (query points excluded from the input mask).
- Add `IrregularForecastDataModule` + `IrregularForecastConfig`: obs_frac timespan split into context window and future targets.
- Add `MIMIC` load-from-file dataset (requires credentialed PhysioNet access).
- Add `UEAIrregular`: wraps any UEA dataset with synthetic timestamp dropout.
- Add `IrregularWrapper`: wraps any `TimeSeriesDataset` with per-window dropout.
- Add `IrregularInterpolationExp` + `IrregularForecastExp` experiment base classes (masked MSE on held-out and future query points).
- Add `GRUDIrregularInterpolation` + `GRUDIrregularForecast` combo experiment classes.
- GRU-D: seq2seq mode — `forward(x, t, mask, t_query) → (B, Tq, F)`.

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
