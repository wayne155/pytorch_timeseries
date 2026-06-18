Architecture
============

The project is moving toward a small set of explicit boundaries:

- datasets describe benchmark data;
- task DataModules prepare batches;
- experiment engines train, validate, test, and checkpoint models;
- result backends store result records;
- artifact backends store downloadable files;
- benchmark reports and leaderboards read stored results.

This separation keeps task/model code independent from reporting and deployment
concerns.

Datasets
--------

Datasets expose raw benchmark data. They may be regular-grid time series,
anomaly datasets, UEA classification datasets, or irregular sample datasets.
Task code should not assume every dataset is a continuous timestamped feature
matrix.

Task DataModules
----------------

Task DataModules adapt datasets to a task. They own splitting, scaling, window
construction, masks, labels, and named batch objects.

Examples:

- ``ForecastDataModule``
- ``ImputationDataModule``
- ``AnomalyDataModule``
- ``UEADataModule``
- ``IrregularClassificationDataModule``
- ``IrregularInterpolationDataModule``
- ``IrregularForecastDataModule``

Named Batches
-------------

New task code should use named batch objects. This avoids fragile positional
tuple unpacking and makes model/task contracts easier to inspect.

Experiment Engine
-----------------

An experiment engine owns the runtime loop:

- initialize dataset and DataModule;
- build the model;
- train and validate each epoch;
- apply early stopping and scheduling;
- load the best checkpoint;
- evaluate on the test split;
- expose hyperparameters, history, metrics, and artifact paths.

The canonical forecast engine currently powers migrated forecast models such as
``DLinear`` and ``Crossformer``.

Task and Model Configuration
----------------------------

The public ``Experiment`` API stays ergonomic and flat:

.. code-block:: python

   Experiment(
       model="DLinear",
       task="Forecast",
       dataset="ETTh1",
       windows=96,
       pred_len=96,
       lr=0.001,
   )

Internally, these settings are split into:

``Task Configuration``
   Settings that define data shape and task semantics.

``Model Configuration``
   Settings that define model architecture.

``Runtime Configuration``
   Settings that define optimization, storage, device, and execution behavior.

Unknown or irrelevant settings fail early.

Results Boundary
----------------

Experiment engines emit ``RunResult`` records. Result backends store those
records. Benchmark reports and leaderboards read stored records and curated
reference entries.

Task and model code should not know about leaderboard rendering, CSV export, or
web dashboards.

Compatibility Shims
-------------------

Legacy experiment names remain importable during migration. They should delegate
to the canonical architecture where possible. New development should target:

- v2 DataModules;
- named batches;
- typed task/model/runtime configuration;
- ``Experiment`` as the high-level entrypoint;
- ``RunResult`` as the result record.
