## 0.2.15

feat: `VanillaTransformer` model — encoder-only Transformer forecaster built from `torch_timeseries.nn` building blocks (`DataEmbedding` + `Encoder`/`EncoderLayer`/`FullAttention`); supports forecasting, anomaly detection, imputation, and classification (`output_prob`); optional RevIN; `VanillaTransformerForecast`, `VanillaTransformerUEAClassification`, `VanillaTransformerAnomalyDetection`, `VanillaTransformerImputation` experiment wrappers; `VanillaTransformerConfig` with head-count divisibility validation; 26 model tests + 8 config tests

feat: `tests/nn/test_multiwavelet.py` — 30 tests for `MultiWaveletTransform` (self-attention) and `MultiWaveletCross` (cross-attention): construction with both `legendre` and `chebyshev` bases, forward shape, finite outputs, gradient flow, padding behaviors (L_q > L_v, L_q < L_v), softmax activation, multi-CZ-layer stacking

feat: leaderboard reproduce scripts for TCN, PatchMixer, RNN and VanillaTransformer in `anomaly_detection/`, `imputation/`, and `uea_classification/` (12 new scripts); VanillaTransformer also in `long_term_forecast/`

test: 828 tests total (+64 vs 0.2.14)

## 0.2.14

feat: `TCNForecaster` model — TCN-based multi-task forecaster (forecast, imputation, anomaly detection, classification) using `TemporalConvNet` with optional RevIN normalization; `TCNForecast`, `TCNUEAClassification`, `TCNAnomalyDetection`, `TCNImputation` experiment wrappers; 15 tests

feat: `PatchMixer` model — patch-based MLP-Mixer forecaster (channel-independent); combines `Patcher` + `MixerBlock` with optional RevIN; all four task wrappers; 14 tests

feat: `torch_timeseries.nn.augmentation` — 6 time-series augmentation modules (`Jitter`, `Scaling`, `Flip`, `WindowCutout`, `MagnitudeWarp`, `TimeWarp`) plus `Compose`, all as `nn.Module` subclasses applied only during `model.training`; 33 tests

feat: `torch_timeseries.nn.Patcher` — patch extraction module `(B, L, C) → (B, N, patch_len, C)` with `'start'`/`'end'`/`'none'` padding modes

feat: `torch_timeseries.nn.FeedForward`, `MixerBlock` — standard Transformer FFN and TSMixer-style time+channel MLP-Mixer block

feat: `torch_timeseries.nn.set_seed` — seeds Python random, NumPy, and PyTorch CPU/CUDA; optional cuDNN determinism flag; 6 tests

test: 721 tests total (+133 vs 0.2.13) — new coverage for `FullAttention`, `AttentionLayer`, `AutoCorrelation`, `AutoCorrelationLayer`, `MovingAvg`, `SeriesDecomp`, `SeriesDecompMulti`, all embedding classes, `ConvLayer`, `EncoderLayer`, `Encoder`, `DecoderLayer`, `Decoder`, `Inception_Block_V1/V2`, `EarlyStopping`, `model_summary`

## 0.2.13

feat: `torch_timeseries.nn.CausalConv1d`, `TemporalBlock`, `TemporalConvNet` — TCN building blocks with causal dilated convolutions and exponential receptive field growth (Bai et al., 2018); 22 tests

fix: `torch_timeseries.experiments.ImputationExp` and `AnomalyDetectionExp` — add `rmse` metric (`MeanSquaredError(squared=False)`) to `_init_metrics()`, matching the pattern already in `ForecastExp` and irregular experiments

fix: `SinusoidalEmbedding` crash with odd `d_model` — cosine assignment now uses `div[:d_model//2]` to avoid size mismatch

feat: `tests/nn/` — 45 tests for `CausalConv1d`/`TemporalBlock`/`TemporalConvNet` and all four temporal encoding modules (`Time2Vec`, `LearnableFourierFeatures`, `RotaryEmbedding`, `SinusoidalEmbedding`)

feat: `FreTSAnomalyDetection`, `FreTSImputation`, `FITSAnomalyDetection`, `FITSImputation` experiment wrappers — completing the four-task coverage for FreTS and FITS

feat: export `EarlyStopping` from `torch_timeseries.utils`; add 19 tests covering patience, delta, state serialization, and checkpoint saving

fix: `ImputationExp` and `AnomalyDetectionExp` `_init_metrics()` — add `rmse: MeanSquaredError(squared=False)` to match ForecastExp

## 0.2.12

feat: `torch_timeseries.augment` — 11 composable time-series augmentation transforms: `Compose`, `Jitter`, `Scale`, `MagnitudeWarp`, `TimeWarp`, `WindowSlice`, `Permute`, `Flip`, `RandomMask`, `RandomApply`, `RandomChoice`; 51 tests

feat: `torch_timeseries.metrics` — `SMAPE`, `MASE`, `QuantileLoss` (torchmetrics-compatible) + `naive_seasonal_mae` helper; 18 tests

feat: `torch_timeseries.scaler` — `MinMaxScaler` (configurable feature range) and `RobustScaler` (median + IQR scaling); 21 tests

feat: `torch_timeseries.utils` — `WarmupCosineScheduler`, `WarmupLinearScheduler` (linear warmup + cosine/linear decay); `model_summary()` returning param counts and size in MB; 12 tests

feat: `torch_timeseries.nn.RevIN` — Reversible Instance Normalization (Kim et al. ICLR 2022) with learnable affine parameters; 13 tests

feat: `torch_timeseries.results.ResultsComparator` — aggregate `RunResult` objects across seeds, compute mean±std per metric, render PrettyTable or pandas DataFrame; 12 tests

feat: comprehensive forecasting model tests — shape + gradient flow for all 17 forecasting models; 18 tests

feat: leaderboard reproduce scripts for `SegRNN`, `TimeMixer`, `TiDE`, `NHiTS` in `long_term_forecast/` and `short_term_forecast/`

docs: `augment.rst`, `metrics.rst` pages added; `scaler.rst` and `nn.rst` updated

fix: `FEDformer` `FourierBlock`/`FourierCrossAttention` hardcoded `n_heads=8` causing einsum shape mismatch when `n_heads != 8`

## 0.2.11
feat: `NHiTS` — Neural Hierarchical Interpolation (Challu et al., AAAI 2023); `NHiTSForecast/Imputation/AnomalyDetection/UEAClassification`

## 0.2.10
feat: `TiDE` — Time-series Dense Encoder (Das et al., TMLR 2023); `TiDEForecast/Imputation/AnomalyDetection/UEAClassification`

## 0.2.9
feat: `TimeMixer` — Past Decomposable Mixing + Future Multipredictor Mixing (Wang et al., ICLR 2024); `TimeMixerForecast/Imputation/AnomalyDetection/UEAClassification`
docs: update model.rst to 27 models + TimeMixer card

## 0.2.8
feat: `SegRNN` — Segment RNN (Lin et al., ICLR 2024); IMO decoding; channel-independent; `SegRNNForecast/Imputation/AnomalyDetection/UEAClassification`
feat: leaderboard reproduce scripts for irregular classification/interpolation/forecast (GRUD, mTAN, LatentODE) and SegRNN forecast
docs: update model.rst to 26 models + SegRNN card + irregular coverage table; add irregular DataModule docs; update README task/model tables
fix: CLI help text lists all 26 models and 9 tasks

## 0.2.7
feat: Irregular TS Phase 3 — mTAN, LatentODE (torchdiffeq), NeuralCDE (torchcde), Raindrop (torch_geometric) models; mTAN classification/interpolation/forecast + LatentODE × 3 + NeuralCDE + Raindrop classification experiments
feat: `pip install torch-timeseries[irregular]` extras group for optional irregular-TS dependencies

## 0.2.6
feat: Irregular TS Phase 2 — `IrregularInterpolationDataModule`, `IrregularForecastDataModule`, `MIMIC` load-from-file dataset, `UEAIrregular` synthetic-dropout wrapper, `IrregularWrapper` for regular datasets
feat: `IrregularInterpolationExp` + `IrregularForecastExp` base experiment classes with masked MSE loss on query points
feat: `GRUDIrregularInterpolation` + `GRUDIrregularForecast` combo experiment classes
model: GRU-D seq2seq mode — `forward(x, t, mask, t_query) → (B, Tq, F)` via hidden-state decay and `fc_seq2seq` projection

## 0.2.5
feat: temporal encoding — `Time2Vec`, `LearnableFourierFeatures`, `RotaryEmbedding`, `SinusoidalEmbedding` added to `torch_timeseries.nn`
docs: model reference section in Sphinx docs with paper citations and Args for all 20 models; nn temporal encoding section
docs: README restructure — new TOC covering 6 time-series tasks; full examples + figures for Imputation, Anomaly Detection, and Classification
refactor: remove duplicate `MovingAvg`/`SeriesDecomp`/`SeriesDecompMulti` class bodies in `Autoformer_EncDec.py` (now imported from `kernels`/`decomp`); fix hard-coded `.cuda()` → `.to(device)` in `AutoCorrelation.py`; remove import shadowing in `nn/__init__.py`; remove dead code in `Crossformer.py`, `DLinear.py`, `embedding.py`, `MultiWaveletCorrelation.py`

## 0.2.4
feat: `NsDiff` — full Non-Stationary DDPM rewrite (local-variance-adapted noise schedule via `betas_tilde`/`betas_bar`; `_NsDenoiser` takes `[y_t ‖ y_0_hat ‖ gx]`; `_SigmaNet` rolling-variance MLP; unconditional generation from `N(0, gx)` prior)
feat: `TMDM` — TMDM-style DDPM with GRU prior-mean network (`_MuNet`); denoiser takes `[y_t ‖ y_0_hat]`; `NsDiffGeneration` / `TMDMGeneration` experiment wrappers
docs: README table of contents; figures for every code example (`forecast_custom_pipeline`, `fast_eval_windows`, `prob_forecast`, `nsdiff_generation`, `experiment_builder`)
rename: `NSDiffusion` → `NsDiff` throughout (model, experiment, leaderboard scripts, tests)

## 0.2.1

feat: add per-task leaderboard reproduce scripts for anomaly detection, imputation, long-term forecast, short-term forecast, and UEA classification across DLinear, NLinear, PatchTST, iTransformer, TimesNet, Autoformer, and FEDformer
feat: add NLinear model and experiment wrappers
feat: add TimesNet imputation and anomaly detection experiment wrappers
update: add README examples for researcher workflows using custom dataloaders, custom training loops, and default experiment registration
update: add default ForecastExp batch handling for simple models that map `batch_x` to forecast predictions
fix: normalize anomaly metric names for leaderboard rendering

## 0.0.3
fixed: dependencies issue of torch-timeserie pypi package 


## 0.1.0

func: Forecast/Imputation/UEAClassification AnomalyDetection experiments
func: pytexp entrypoint


## 0.1.1
fix: fix a typo of the standard scaler
fix: fix some cli errors

## 0.1.2
func: add Informer Forecast/Imputation/AnomalyDetection/UEAClassification
func: add Autoformer Forecast/Imputation/AnomalyDetection/UEAClassification
func: add FEDformer Forecast/Imputation/AnomalyDetection/UEAClassification
func: add PatchTST Forecast/Imputation/AnomalyDetection/UEAClassification

## 0.1.3

func: add iTransformer Forecast/Imputation/AnomalyDetection/UEAClassification
fix: fix a typo of Informer d_layer config


## 0.1.6

func: adding a new model CATS
func: adding year into time features
func: add popular and set as default split, ETT for 6:2:2, others for 7:1:2
fix: default using only train data to scale 


## 0.1.7

func: we make the default dataloader settings identical with Time-Series-Libary



## 0.1.8

func: add new dataset wrapper, MultivariateFast, to split by window not by steps.

## 0.1.9

func: add a nonoverlap dataloader
fix: change default time encoding to 3, for stable data range in 0~1. We found that data like 30, 2024 will cause unstable training or even corrupted training process.


## 0.1.10

fix: fix data loading bugs


## 0.1.12

update: update iTransformer default configurations
update: update Forecast default parameter (l2_decay=0, lr=0.0001)
update: update timeenc config (timeenc=0)

## 0.1.13
bug fixed
## 0.1.14

update: change ident to seed+md5

## 0.1.15

update: new dataloader, sliding window with time index

## 0.1.16

update: deprecate slidingWindowTimeIndex class, include a extra argument timeindex in the SlidingWindowTS class

## 0.1.17
update: adding ReConstruct Dataset and Dataloader


## 0.1.18
update: remove the function config_wandb name parameters in UEAClassification

## 0.1.19
add: add model FITS, UEAClassification and Forecast


## 0.1.20
fix some bugs



## 0.1.22
fix bug in Electricity

## 0.1.23
add sim freq dataset, embedding back to previous version.



## 0.1.24
ILI freq from yh to h

## 0.2.0
feat: `time_enc` in v2 dataloader configs now accepts readable string aliases (`"calendar"`, `"fourier"`, `"normalized"`) in addition to existing `int` and `TimeEncoding` enum values — applies to `WindowConfig`, `ImputationWindowConfig`, and `IrregularClassificationConfig`

**BREAKING**: default value of `time_enc` changed from `0` (integer) to `"calendar"` (string) — code comparing `config.time_enc == 0` must be updated

## 0.2.1
fix: ETT datasets now use canonical calendar splits (12/4/4 months, TSLib borders) instead of 7:1:2 ratios; `SplitConfig` gains explicit `borders` and dataset-default splits (`DEFAULT_SPLIT_CONFIGS`)
fix: training protocol aligned with Time-Series-Library — `lradj=type1` (lr halved per epoch), 10 epochs, patience 3, no gradient clipping; DLinear restored moving-average weight init
refactor: `ForecastConfig` removed; engines take `WindowConfig` + optional `SplitConfig`; v2 dataloader configs split into `window.py` / `split.py` / `loader.py`; `scale_in_train` removed (scaler always fits on train)
feat: `LeaderboardExperiment` + task/model reproduce scripts (`leaderboard/reproduce/forecast/*.py`) with multi-GPU lane parallelism; leaderboard webapp redesign (moved to `webapp/leaderboard/`)

## 0.2.2
feat: `build_dataset(csv=..., freq=...)` builds a dataset directly from a local CSV; `num_features`/`length` are now inferred from loaded data instead of class attributes
feat: default dataset directory is now `~/.torchtimeseries/data` (override with `root=` / `data_path=`)
refactor: torchvision dependency removed — download/extract/integrity utilities extracted to `torch_timeseries/dataset/utils.py`
docs: README section on custom datasets; compact format for curated leaderboard entries

## 0.2.3
feat: probabilistic forecasting — `ProbForecastExp` (model returns its own training loss; inference returns `(batch, pred_len, num_features, samples)` ensembles; early stopping on val CRPS)
feat: probabilistic metrics package `torch_timeseries.metrics` — CRPS (native vectorized estimator, no external deps), CRPSSum, QICE, PICP, ProbMSE/ProbMAE/ProbRMSE
feat: `WindowConfig.fast_val` / `fast_test` documented — sliding-window training with non-overlapping val/test windows to accelerate expensive (e.g. diffusion) inference
refactor: `_process_one_batch(batch: TSBatch)` — forecast experiments receive the structured batch instead of six positional tensors; tuple unpacking removed from train/eval loops
