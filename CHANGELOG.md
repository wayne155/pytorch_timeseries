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
