Creating Datasets
=================

There are two paths: a one-liner for local CSV files, and a full subclass for
datasets that require downloading or non-trivial preprocessing.

From a CSV File
---------------

The quickest way to bring your own data is ``build_dataset``. The CSV must have
a ``date`` column; every other column becomes a feature.

.. code-block:: python

   from torch_timeseries.dataset import build_dataset

   dataset = build_dataset(csv="./my_data.csv", freq="h")
   print(dataset.num_features)   # number of feature columns
   print(dataset.data.shape)     # (T, num_features)

The ``freq`` argument uses pandas frequency aliases (``"h"`` for hourly,
``"t"`` for minutely, ``"d"`` for daily, etc.).

The returned object is a fully functional ``TimeSeriesDataset`` — pass it
directly to any DataModule:

.. code-block:: python

   from torch_timeseries.scaler import StandardScaler
   from torch_timeseries.dataloader.v2 import ForecastDataModule, WindowConfig

   dm = ForecastDataModule(
       dataset=build_dataset(csv="./my_data.csv", freq="h"),
       scaler=StandardScaler(),
       window=WindowConfig(window=96, horizon=1, steps=96),
   )

Subclassing TimeSeriesDataset
------------------------------

For datasets that need downloading, custom parsing, or a canonical benchmark
split, subclass ``TimeSeriesDataset`` and implement two methods.

.. code-block:: python

   import os
   import numpy as np
   import pandas as pd
   from torch_timeseries.core import TimeSeriesDataset, Freq

   class MySensors(TimeSeriesDataset):
       name: str = "MySensors"   # cache subdirectory under the data root
       freq: Freq = "h"

       def download(self):
           # Fetch raw files into self.dir, or no-op if already local.
           pass

       def _load(self) -> np.ndarray:
           path = os.path.join(self.dir, "sensors.csv")
           df = pd.read_csv(path, parse_dates=["date"])
           self.df    = df
           self.dates = pd.DataFrame({"date": df["date"]})
           self.data  = df.drop("date", axis=1).to_numpy()   # (T, C)
           return self.data

The contract is deliberately small:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Attribute to set
     - Type and meaning
   * - ``self.df``
     - ``pandas.DataFrame`` with at least a ``date`` column.
   * - ``self.dates``
     - ``pandas.DataFrame`` with a single ``date`` column aligned to the data rows.
   * - ``self.data``
     - ``numpy.ndarray`` of shape ``(T, num_features)`` — the numeric features.

``num_features`` and ``length`` are derived from ``self.data`` automatically.

Registering a Canonical Split
------------------------------

Built-in datasets ship a canonical train/val/test split that DataModules use
when no explicit ``SplitConfig`` is provided. Register your own:

.. code-block:: python

   from torch_timeseries.dataloader.v2.split import DEFAULT_SPLIT_CONFIGS
   from torch_timeseries.dataloader.v2 import SplitConfig

   DEFAULT_SPLIT_CONFIGS["MySensors"] = SplitConfig(
       borders=(700, 900, 1000)   # train ends at 700, val at 900, test at 1000
   )

After registration ``ForecastDataModule(dataset=MySensors())`` uses the
academic split automatically — no ``split=`` argument needed.
