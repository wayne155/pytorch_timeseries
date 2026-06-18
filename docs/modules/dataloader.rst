torch_timeseries.dataloader
============================

DataModules wrap datasets into task-specific train/val/test loaders. The v2
family (importable from ``torch_timeseries.dataloader.v2``) is the recommended
API for new code — it uses named batch objects and strict configuration
validation.

.. code-block:: python

   from torch_timeseries.dataset import ETTh1
   from torch_timeseries.scaler import StandardScaler
   from torch_timeseries.dataloader.v2 import (
       ForecastDataModule,
       WindowConfig,
       SplitConfig,
       LoaderConfig,
   )

   dm = ForecastDataModule(
       dataset=ETTh1(),
       scaler=StandardScaler(),
       window=WindowConfig(window=96, horizon=1, steps=96),
       split=SplitConfig(train=0.7, val=0.1, test=0.2),
       loader=LoaderConfig(batch_size=32),
   )

   batch = next(iter(dm.train_loader))
   print(batch.x.shape)   # (32, 96, num_features)
   print(batch.y.shape)   # (32, 96, num_features)

All five task DataModules follow the same shape. See
:doc:`../concepts/dataloader` for the full batch field reference.

----

v2 DataModules
--------------

The canonical API for new code. Each DataModule is task-specific and yields
named batches.

.. currentmodule:: torch_timeseries.dataloader.v2

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   ForecastDataModule
   ImputationDataModule
   AnomalyDataModule
   UEADataModule
   IrregularClassificationDataModule
   IrregularInterpolationDataModule
   IrregularForecastDataModule

Configuration Objects
---------------------

Passed to DataModule constructors to configure windowing, splitting, and
loading behaviour.

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   WindowConfig
   SplitConfig
   LoaderConfig
   UEAWindowConfig
   IrregularClassificationConfig
   IrregularInterpolationConfig
   IrregularForecastConfig

Legacy Dataloaders
------------------

These classes pre-date the v2 API and remain available for compatibility.
New code should use v2 DataModules.

Forecast
~~~~~~~~

.. currentmodule:: torch_timeseries.dataloader

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_timeseries.dataloader.forecast_loaders %}
     {{ name }}
   {% endfor %}

Imputation
~~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_timeseries.dataloader.imputation_loaders %}
     {{ name }}
   {% endfor %}

Classification
~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_timeseries.dataloader.classification_loaders %}
     {{ name }}
   {% endfor %}

Anomaly Detection
~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_timeseries.dataloader.anomaly_loaders %}
     {{ name }}
   {% endfor %}
