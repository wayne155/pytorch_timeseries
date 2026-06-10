Changelog
=========

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

For older release notes, see the project root ``CHANGELOG.md``.
