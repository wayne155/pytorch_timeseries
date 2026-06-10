

DataModules and Batches
=======================

The v2 dataloader API is the canonical data preparation layer for new code.
It turns a dataset into train, validation, and test loaders for a specific
task. Each loader yields a named batch object so training code can use explicit
fields instead of positional tuple indexes.

Why v2 Exists
-------------

The legacy dataloaders are still importable for compatibility, but new task and
model work should use v2 DataModules. The important differences are:

- task-specific configuration is explicit and validated;
- batches have names such as ``batch.x`` and ``batch.y``;
- time features, raw values, masks, and labels are represented by batch fields;
- data splitting and scaling are owned by the DataModule.

Forecast DataModule
-------------------

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
       window=WindowConfig(
           window=96,
           horizon=1,
           steps=96,
           include_raw=True,
           include_time=True,
       ),
       split=SplitConfig(train=0.7, val=0.1, test=0.2),
       loader=LoaderConfig(batch_size=32, shuffle_train=True),
   )

   batch = next(iter(datamodule.train_loader))
   print(batch.x.shape)
   print(batch.y.shape)

Common Forecast Batch Fields
----------------------------

``batch.x``
   Scaled input window.

``batch.y``
   Scaled target window.

``batch.x_raw`` and ``batch.y_raw``
   Original unscaled values, available when ``include_raw=True``.

``batch.x_time`` and ``batch.y_time``
   Calendar/time features, available when ``include_time=True``.

Column Selection
----------------

Forecast windows can select input and target columns by index or by name. This
is useful for multivariate datasets where the model should observe one set of
features and predict another.

.. code-block:: python

   WindowConfig(
       window=96,
       horizon=1,
       steps=96,
       input_columns=["HUFL", "HULL"],
       target_columns=["OT"],
   )

Supported v2 Families
---------------------

The package includes v2 DataModule families for:

- forecasting;
- imputation;
- anomaly detection;
- UEA classification;
- irregular classification.

The public shape is intentionally similar across tasks: create a task-specific
window/config object, pass split and loader settings, then iterate named
batches.
