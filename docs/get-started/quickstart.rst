Quickstart
==========

``torch_timeseries`` is a research toolkit for time-series experiments. You can
either use the dataset/DataModule layer with your own training loop, or register
models into the default experiment runner.

Install
-------

Install the package after installing a PyTorch build that matches your machine:

.. code-block:: bash

   pip install torch-timeseries

For development from this repository:

.. code-block:: bash

   pip install -r requirements.txt

Two Ways to Use
---------------

Way 1 - Custom pipeline
~~~~~~~~~~~~~~~~~~~~~~~

Import a dataset and DataModule, then write your own training logic. This gives
you full control over the loss, optimizer, and batch handling.

.. code-block:: python

   import torch
   import torch.nn as nn

   from torch_timeseries.dataset import ETTh1
   from torch_timeseries.scaler import StandardScaler
   from torch_timeseries.dataloader.v2 import (
       ForecastDataModule,
       LoaderConfig,
       SplitConfig,
       WindowConfig,
   )

   # Dataset is downloaded automatically on first use.
   dataset = ETTh1("./data")

   dm = ForecastDataModule(
       dataset=dataset,
       scaler=StandardScaler(),
       window=WindowConfig(window=96, horizon=1, steps=96),
       # ETTh1 academic split: 12 months train, 4 months val, 4 months test.
       # If split is omitted, the datamodule uses this dataset default.
       split=SplitConfig(borders=(12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24)),
       loader=LoaderConfig(batch_size=32),
   )

   class LinearForecaster(nn.Module):
       """Input: (batch, input_window, features). Output: (batch, pred_len, features)."""

       def __init__(self, input_window: int, pred_len: int):
           super().__init__()
           self.proj = nn.Linear(input_window, pred_len)

       def forward(self, x):
           # x: (B, 96, C) -> (B, C, 96) -> (B, C, 96) -> (B, 96, C)
           return self.proj(x.transpose(1, 2)).transpose(1, 2)

   device = "cuda" if torch.cuda.is_available() else "cpu"
   model = LinearForecaster(input_window=96, pred_len=96).to(device)
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   loss_fn = nn.MSELoss()

   for epoch in range(1):
       model.train()
       for batch in dm.train_loader:
           # Each batch is a TSBatch.
           x = batch.x.float().to(device)  # (B, 96, num_features)
           y = batch.y.float().to(device)  # (B, 96, num_features)

           optimizer.zero_grad()
           pred = model(x)                 # (B, 96, num_features)
           loss = loss_fn(pred, y)
           loss.backward()
           optimizer.step()

Use this pattern when you need a non-standard training loop, custom loss, or
are prototyping a new architecture.

Way 2 - Default training paradigm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the built-in experiment runner. Pick a model, task, and dataset; the library
handles data loading, training, evaluation, and result saving.

This path works for built-in models and for your own models registered with the
default experiment classes.

Architecture Direction
^^^^^^^^^^^^^^^^^^^^^^

New development targets the v2 DataModule API and the high-level
:class:`torch_timeseries.Experiment` entrypoint. Legacy dataloaders and direct
experiment classes remain available for compatibility, but new task/model
features should use named batches, task DataModules, and result records.

Register Custom Models
^^^^^^^^^^^^^^^^^^^^^^

To use the default training loop with your own model, subclass the task
experiment class, define ``_init_model``, then register it.

For forecasting, the model should read ``batch_x`` with shape
``(batch, windows, num_features)`` and return predictions with shape
``(batch, pred_len, num_features)``.

.. code-block:: python

   from dataclasses import dataclass

   import torch
   import torch.nn as nn

   from torch_timeseries import Experiment, register_model
   from torch_timeseries.experiments import ForecastExp


   class MyForecastNet(nn.Module):
       """Input: (B, seq_len, C). Output: (B, pred_len, C)."""

       def __init__(self, seq_len: int, pred_len: int):
           super().__init__()
           self.proj = nn.Linear(seq_len, pred_len)

       def forward(self, x):
           return self.proj(x.transpose(1, 2)).transpose(1, 2)


   @dataclass
   class MyForecastModel(ForecastExp):
       model_type: str = "MyForecastModel"

       def _init_model(self):
           self.model = MyForecastNet(
               seq_len=self.windows,
               pred_len=self.pred_len,
           ).to(self.device)


   register_model(MyForecastModel)

   # The registered model name is the class name.
   device = "cuda" if torch.cuda.is_available() else "cpu"

   results = Experiment(
       model="MyForecastModel",
       task="Forecast",
       dataset="ETTh1",
       windows=96,
       pred_len=96,
       epochs=1,
       device=device,
   ).run(seeds=[1])

   print(results[0].metrics)

The same registered model can be launched from the CLI after the Python module
containing ``register_model(...)`` has been imported:

.. code-block:: bash

   pytexp --model MyForecastModel --task Forecast --dataset_type ETTh1 run 1

Run Built-In Models
^^^^^^^^^^^^^^^^^^^

Experiment builder:

.. code-block:: python

   from torch_timeseries import Experiment

   # Single run: returns a RunResult with metrics, hparams, git commit, and timing.
   result = Experiment(model="DLinear", task="Forecast", dataset="ETTh1").run(seeds=[1])
   print(result[0].metrics)

   # Multiple seeds, save results to disk.
   results = Experiment(
       model="DLinear",
       task="Forecast",
       dataset="ETTh1",
       windows=96,
       pred_len=96,
       lr=0.001,
       save_dir="./results",
   ).run(seeds=[1, 2, 3])

   # Grid search across models and datasets.
   Experiment.grid(
       models=["DLinear", "Autoformer"],
       tasks=["Forecast"],
       datasets=["ETTh1", "ETTm1"],
       seeds=[1, 2, 3],
       save_dir="./results",
   ).run()

   # Compare saved results.
   Experiment.compare(save_dir="./results", task="Forecast")

CLI:

.. code-block:: bash

   # forecast
   pytexp --model DLinear --task Forecast --dataset_type ETTh1 run 3
   pytexp --model DLinear --task Forecast --dataset_type ETTh1 runs '[1,2,3]'

   # imputation
   pytexp --model DLinear --task Imputation --dataset_type ETTh1 run 3

   # anomaly detection
   pytexp --model DLinear --task AnomalyDetection --dataset_type MSL run 3

   # classification
   pytexp --model DLinear --task UEAClassification --dataset_type EthanolConcentration run 3

   # compare saved results
   pytexp compare --save_dir ./results --task Forecast

Next Steps
----------

- Read :doc:`/concepts/experiments` for the experiment workflow.
- Read :doc:`/concepts/results-and-artifacts` for result storage and model downloads.
- Read :doc:`/concepts/dataloader` for v2 DataModules and named batches.
