Results and Artifacts
=====================

Every experiment run produces a :class:`RunResult`. When a local result backend
is configured, the result record is written to disk and model checkpoints are
registered as downloadable artifacts.

Enable Local Storage
--------------------

Pass ``save_dir`` directly:

.. code-block:: python

   from torch_timeseries import Experiment

   results = Experiment(
       model="DLinear",
       task="Forecast",
       dataset="ETTh1",
       windows=96,
       pred_len=96,
       save_dir="./results",
   ).run(seeds=[1])

or attach a backend explicitly:

.. code-block:: python

   Experiment(
       model="DLinear",
       task="Forecast",
       dataset="ETTh1",
       windows=96,
       pred_len=96,
   ).with_local("./results").run(seeds=[1])

RunResult Fields
----------------

Important fields include:

``metrics``
   Final test metrics, such as ``mse`` and ``mae`` for forecasting.

``history``
   Per-epoch training loss and validation metrics when the engine records them.

``hparams``
   The full hyperparameter snapshot saved with the run.

``run_config``
   The normalized result-affecting configuration used to identify equivalent
   experiment setups.

``config_hash``
   A stable fingerprint of ``run_config``. It excludes seed and storage
   metadata.

``run_id``
   A seeded run identity, currently shaped as ``seed{N}-{config_hash}``.

``artifacts``
   Downloadable files produced by the run, such as the best model checkpoint.

Storage Layout
--------------

Result records are grouped by configuration:

.. code-block:: text

   results/
     records/
       DLinear/
         Forecast/
           ETTh1/
             {config_hash}/
               config.json
               seed1.json
               seed2.json

Model artifacts are stored separately:

.. code-block:: text

   results/
     artifacts/
       DLinear/
         Forecast/
           ETTh1/
             {config_hash}/
               seed1/
                 best_model.pth

The training engine may also keep its own run directory:

.. code-block:: text

   results/
     runs/
       DLinear/
         Forecast/
           ETTh1/
             {config_hash}/
               seed1-{config_hash}/
                 best_model.pth

Why Hash Configurations?
------------------------

Older result filenames were based on model, task, dataset, and seed. That meant
two different configurations with the same seed could overwrite each other.

The current layout uses ``config_hash`` so different settings are stored
independently and the leaderboard can compare arbitrary configurations instead
of only a few hard-coded columns.

What Goes Into the Hash?
------------------------

The hash is built from result-affecting configuration:

- model, task, and dataset;
- task settings such as ``windows``, ``pred_len``, ``horizon``, masks, and
  splits;
- model settings such as architecture dimensions;
- training settings such as ``lr``, ``batch_size``, ``epochs``, loss, and
  scaler.

It excludes storage and infrastructure settings such as ``save_dir``, ``device``,
``num_worker``, and ``pin_memory``.

Artifact Backends
-----------------

The local artifact backend is implemented today. Database, object storage, or
model-hub backends can implement the same artifact backend interface and attach
downloadable model locations to ``RunResult.artifacts`` without changing the
experiment workflow.
