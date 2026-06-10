Quickstart
==========

``torch_timeseries`` is a research toolkit for time-series experiments. It can
either run standard benchmarks for you or provide datasets and DataModules for
your own training loop.

Install
-------

Install the package after installing a PyTorch build that matches your machine:

.. code-block:: bash

   pip install torch-timeseries

For development from this repository:

.. code-block:: bash

   pip install -r requirements.txt

Run a Built-In Experiment
-------------------------

The fastest path is the high-level :class:`torch_timeseries.Experiment` API.
Choose a model, task, dataset, and any configuration overrides.

.. code-block:: python

   from torch_timeseries import Experiment

   results = Experiment(
       model="DLinear",
       task="Forecast",
       dataset="ETTh1",
       windows=96,
       pred_len=96,
       lr=0.001,
       save_dir="./results",
   ).run(seeds=[1, 2, 3])

   for result in results:
       print(result.seed, result.metrics)

The runner prepares the dataset, builds the model, trains, validates, tests, and
stores a :class:`torch_timeseries.results.RunResult`.

Use the CLI
-----------

The CLI exposes the same experiment family:

.. code-block:: bash

   pytexp --model DLinear --task Forecast --dataset_type ETTh1 run 1
   pytexp --model DLinear --task Forecast --dataset_type ETTh1 runs '[1,2,3]'

Compare saved local results:

.. code-block:: bash

   pytexp compare --save_dir ./results --task Forecast

Write Your Own Training Loop
----------------------------

Use v2 DataModules when you want full control over optimization or model code.
They return named batch objects instead of positional tuples.

.. code-block:: python

   from torch_timeseries.dataset import ETTh1
   from torch_timeseries.scaler import StandardScaler
   from torch_timeseries.dataloader.v2 import (
       ForecastDataModule,
       LoaderConfig,
       SplitConfig,
       WindowConfig,
   )

   dataset = ETTh1("./data")
   datamodule = ForecastDataModule(
       dataset=dataset,
       scaler=StandardScaler(),
       window=WindowConfig(window=96, horizon=1, steps=96),
       split=SplitConfig(train=0.7, val=0.1, test=0.2),
       loader=LoaderConfig(batch_size=32),
   )

   for batch in datamodule.train_loader:
       x = batch.x.float()
       y = batch.y.float()
       # your model, loss, optimizer, and logging here

Next Steps
----------

- Read :doc:`/concepts/experiments` for the experiment workflow.
- Read :doc:`/concepts/results-and-artifacts` for result storage and model downloads.
- Read :doc:`/concepts/dataloader` for v2 DataModules and named batches.
