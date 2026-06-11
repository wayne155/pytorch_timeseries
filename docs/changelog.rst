Changelog
=========

0.2.3
-----

- Add probabilistic forecasting via ``ProbForecastExp``.
- Add probabilistic metrics in ``torch_timeseries.metrics``: CRPS, CRPSSum,
  QICE, PICP, ProbMSE, ProbMAE, and ProbRMSE.
- Document ``WindowConfig.fast_val`` and ``fast_test`` for non-overlapping
  validation/test windows.
- Refactor forecast batch handling to structured ``TSBatch`` inputs.

0.2.2
-----

- Add ``build_dataset(csv=..., freq=...)`` for local CSV datasets.
- Infer dataset ``num_features`` and ``length`` from loaded data.
- Remove the runtime ``torchvision`` dependency.
- Add README guidance for custom datasets and compact curated leaderboard
  entries.

0.2.1
-----

- Add per-task leaderboard reproduce scripts for anomaly detection,
  imputation, long-term forecast, short-term forecast, and UEA classification
  across DLinear, NLinear, PatchTST, iTransformer, TimesNet, Autoformer, and
  FEDformer.
- Add NLinear model and experiment wrappers.
- Add TimesNet imputation and anomaly detection experiment wrappers.
- Add README examples for researcher workflows using custom dataloaders,
  custom training loops, and default experiment registration.
- Add default ForecastExp batch handling for simple models that map
  ``batch_x`` to forecast predictions.
- Normalize anomaly metric names for leaderboard rendering.
