Experiments
===========

Experiments are the high-level path for reproducible benchmark runs. They
connect a model, task, dataset, configuration, result backend, and artifact
backend.

Basic Usage
-----------

.. code-block:: python

   from torch_timeseries import Experiment

   results = Experiment(
       model="DLinear",
       task="Forecast",
       dataset="ETTh1",
       windows=96,
       pred_len=96,
       save_dir="./results",
   ).run(seeds=[1, 2, 3])

   print(results[0].metrics)

The public API is flat on purpose: users can pass task, model, and runtime
settings in one constructor. Internally, settings are split into validated task,
model, and runtime configuration objects before data or model construction.

Supported Task Names
--------------------

The standard task names are:

- ``Forecast``
- ``Imputation``
- ``AnomalyDetection``
- ``UEAClassification``

The canonical engine path is being migrated model by model. Migrated forecast
models use the v2 DataModule and typed configuration path. Legacy experiment
classes remain available through compatibility shims while the migration
continues.

Multiple Seeds
--------------

Pass multiple seeds to produce multiple :class:`RunResult` records under the
same run configuration:

.. code-block:: python

   Experiment(
       model="DLinear",
       task="Forecast",
       dataset="ETTh1",
       windows=96,
       pred_len=96,
       save_dir="./results",
   ).run(seeds=[1, 2, 3])

The random seed is not part of the configuration fingerprint. This lets
leaderboards aggregate repeated runs while still preserving every seeded run as
its own result record and model artifact.

Grid Runs
---------

Use :meth:`Experiment.grid` to run repeated combinations:

.. code-block:: python

   Experiment.grid(
       models=["DLinear", "Crossformer"],
       tasks=["Forecast"],
       datasets=["ETTh1", "ETTm1"],
       seeds=[1, 2, 3],
       save_dir="./results",
       windows=96,
       pred_len=96,
   ).run()

Compare Results
---------------

Local results can be compared from Python:

.. code-block:: python

   Experiment.compare(save_dir="./results", task="Forecast")

or from the CLI:

.. code-block:: bash

   pytexp compare --save_dir ./results --task Forecast

Configuration Validation
------------------------

Experiment configuration is strict. Unknown or irrelevant keys fail before the
dataset or model is built. This protects benchmark results from accidental
settings that look meaningful but are not consumed by the selected model or
task.

For example, a DLinear forecast run accepts ``individual`` but rejects
Crossformer-only settings such as ``d_model``.
