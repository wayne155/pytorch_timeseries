# Architecture Roadmap

This roadmap captures the agreed migration path toward the canonical architecture described in `CONTEXT.md` and `docs/adr/`.

## Priorities

1. Define formal configuration objects:
   - Task Configuration: forecast, imputation, anomaly detection, UEA classification, irregular classification.
   - Model Configuration: one model-owned config per migrated model.
   - Runtime and result configuration for engine/device/checkpoint/result behavior.

2. Build the canonical Experiment Engine:
   - Own one training, validation, test, checkpoint, logging, and result-emission loop.
   - Accept a Task Contract, Model Contract, Task Configuration, Model Configuration, and runtime/result configuration.

3. Migrate one vertical slice first:
   - `DLinear` + `Forecast`
   - `ForecastDataModule`
   - `RunResult`
   - Existing `DLinearForecast` name remains as a Compatibility Shim.

4. Repeat migration after the first slice is proven:
   - Remaining Forecast models.
   - Imputation.
   - Anomaly detection.
   - UEA classification.
   - Irregular classification.

5. Deprecate Legacy Loaders:
   - Keep them importable during the transition.
   - Do not add new features to them.
   - Document their future removal.
