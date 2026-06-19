Introduction
============

``torch-timeseries`` is an open-source deep learning library for time series
research built on top of PyTorch. It provides 86+ state-of-the-art model
implementations together with a standardised data pipeline, a high-level
:doc:`Forecaster API <forecaster>`, and a one-command experiment runner,
covering **nine tasks** out of the box:

- **Forecasting** — predict future values from a look-back window
- **Probabilistic Forecasting** — predict with calibrated uncertainty intervals
- **Imputation** — reconstruct missing values under random or block masking
- **Anomaly Detection** — flag unusual timesteps from reconstruction error
- **Classification** — label multivariate sequences (UEA archive)
- **Generation** — synthesise new realistic sequences via diffusion or GANs
- **Irregular Classification** — classify asynchronously sampled sequences
- **Irregular Interpolation** — reconstruct values at arbitrary query times
- **Irregular Forecasting** — forecast from irregular observation schedules

`Source code on GitHub <https://github.com/wayne155/pytorch_timeseries>`__ ·
`API Reference <../modules/model.html>`__

----

Citing
------

If you use this library in academic work, please cite:

.. code-block:: bibtex

   @software{pytorch_timeseries,
     author  = {Ye, Weiwei},
     title   = {torch-timeseries: A Deep Learning Library for Time Series Research},
     url     = {https://github.com/wayne155/pytorch_timeseries},
     year    = {2024},
   }

----

Data Structures
---------------

Datasets
~~~~~~~~

Every built-in dataset is a subclass of ``TimeSeriesDataset``. It exposes three
attributes after loading:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Attribute
     - Description
   * - ``dataset.data``
     - ``numpy.ndarray`` of shape ``(T, num_features)`` — the raw time series.
   * - ``dataset.dates``
     - ``pandas.DataFrame`` with a single ``date`` column aligned to ``data``.
   * - ``dataset.num_features``
     - Number of feature columns (int).

Built-in datasets (ETTh1, ETTm1, MSL, PhysioNet, …) download and cache
automatically on first use. A local CSV can be wrapped in one line:

.. code-block:: python

   from torch_timeseries.dataset import build_dataset

   dataset = build_dataset(csv="sensors.csv", freq="h")
   # CSV must have a 'date' column; every other column becomes a feature.
   print(dataset.num_features)   # number of non-date columns
   print(dataset.data.shape)     # (T, num_features)

See :doc:`create_dataset` for the full subclassing contract.

Named Batches
~~~~~~~~~~~~~

DataModules yield named batch objects — no more positional indexing of tuples.
The fields available depend on the task:

**Forecast batch** (``ForecastDataModule``)

.. code-block:: python

   batch.x          # (B, window,  C) — scaled input
   batch.y          # (B, pred_len, C) — scaled target
   batch.x_raw      # unscaled input   (if include_raw=True)
   batch.x_time     # time features    (if include_time=True)

**Imputation batch** (``ImputationDataModule``)

.. code-block:: python

   batch.x          # (B, T, C) — masked input (zeros at masked positions)
   batch.y          # (B, T, C) — full original window
   batch.mask       # (B, T, C) — 1 = observed, 0 = masked

**Anomaly detection batch** (``AnomalyDataModule``)

.. code-block:: python

   batch.x          # (B, T, C) — input window
   batch.y          # (B, T) — ground-truth anomaly labels

**Classification batch** (``UEAClassificationDataModule``)

.. code-block:: python

   batch.x          # (B, T, C) — padded multivariate sequence
   batch.y          # (B,) — integer class label
   batch.mask       # (B, T) — 1 = valid timestep, 0 = padding

DataModules
~~~~~~~~~~~

All DataModules share the same constructor shape:

.. code-block:: python

   from torch_timeseries.dataloader.v2 import (
       ForecastDataModule, WindowConfig, SplitConfig, LoaderConfig
   )

   dm = ForecastDataModule(
       dataset  = dataset,
       scaler   = StandardScaler(),
       window   = WindowConfig(window=96, horizon=1, steps=96),
       split    = SplitConfig(train=0.7, val=0.1, test=0.2),
       loader   = LoaderConfig(batch_size=32),
   )

   for batch in dm.train_loader:
       ...   # batch.x, batch.y

Signal Splitting
~~~~~~~~~~~~~~~~

``SplitConfig`` accepts either a ratio split or absolute border indices:

.. code-block:: python

   # Ratio split (default 70/10/20)
   SplitConfig(train=0.7, val=0.1, test=0.2)

   # Absolute borders — e.g. ETTh1 canonical split
   SplitConfig(borders=(12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24))

All built-in datasets ship a canonical split registered in
``torch_timeseries.dataloader.v2.split.DEFAULT_SPLIT_CONFIGS``; pass no
``split`` argument to use it.

----

Applications
------------

Sensor Forecasting
~~~~~~~~~~~~~~~~~~

Predict the next 24 hours from 4 days of hourly sensor readings stored in a
local CSV file.

.. code-block:: text

   # sensors.csv
   date,        temp,  pressure, humidity
   2023-01-01,  12.3,  1012.1,   65.1
   2023-01-02,  13.1,  1009.4,   67.4
   ...

.. code-block:: python

   import torch
   import torch.nn as nn
   from torch_timeseries.dataset import build_dataset
   from torch_timeseries.scaler import StandardScaler
   from torch_timeseries.dataloader.v2 import ForecastDataModule, WindowConfig
   from torch_timeseries.model import PatchTST

   dataset = build_dataset(csv="sensors.csv", freq="h")

   dm = ForecastDataModule(
       dataset=dataset,
       scaler=StandardScaler(),
       window=WindowConfig(window=96, horizon=1, steps=24),
   )

   device = "cuda" if torch.cuda.is_available() else "cpu"
   model  = PatchTST(
       seq_len=96, pred_len=24,
       enc_in=dataset.num_features, d_model=64, n_heads=4, e_layers=2,
   ).to(device)
   opt = torch.optim.Adam(model.parameters(), lr=1e-3)

   for epoch in range(20):
       model.train()
       for batch in dm.train_loader:
           x = batch.x.float().to(device)
           y = batch.y.float().to(device)
           opt.zero_grad()
           nn.MSELoss()(model(x), y).backward()
           opt.step()

   model.eval()
   with torch.no_grad():
       batch = next(iter(dm.test_loader))
       preds = model(batch.x.float().to(device))   # (B, 24, 3)

For a full benchmark using the built-in experiment runner:

.. code-block:: python

   from torch_timeseries import Experiment

   results = Experiment(
       model="PatchTST", task="Forecast", dataset="ETTh1",
       windows=96, pred_len=24,
   ).run(seeds=[1, 2, 3])

   print(results[0].metrics)   # {'mse': ..., 'mae': ...}

Anomaly Detection in Industrial Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train a reconstruction model on the MSL (Mars Science Laboratory) dataset and
flag timesteps with high reconstruction error.

.. code-block:: python

   import torch
   import torch.nn as nn
   from dataclasses import dataclass
   from torch_timeseries.experiments import AnomalyDetectionExp

   class LinearReconstructor(nn.Module):
       def __init__(self, seq_len, n_features):
           super().__init__()
           self.proj = nn.Linear(seq_len, seq_len)

       def forward(self, x):   # (B, T, C) → (B, T, C)
           return self.proj(x.transpose(1, 2)).transpose(1, 2)

   @dataclass
   class MyAnomalyExp(AnomalyDetectionExp):
       model_type: str = "LinearReconstructor"

       def _init_model(self):
           self.model = LinearReconstructor(
               self.windows, self.dataset.num_features
           ).to(self.device)

       def _process_one_batch(self, batch_x, origin_x, batch_y):
           x = batch_x.to(self.device, dtype=torch.float32)
           return self.model(x), x     # (pred, true)

   result = MyAnomalyExp(
       dataset_type="MSL", windows=100, anomaly_ratio=0.25,
       epochs=30, device="cuda",
   ).run(seed=1)
   print(result)   # {'precision': ..., 'recall': ..., 'f1': ...}

Or with a built-in model in one line:

.. code-block:: python

   from torch_timeseries import Experiment

   Experiment(model="TimesNet", task="AnomalyDetection",
              dataset="MSL", windows=100).run(seeds=[1, 2, 3])

Multivariate Time Series Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classify multivariate sequences from the `UEA archive
<https://www.timeseriesclassification.com/>`__. Any UEA dataset name is
accepted — it downloads automatically.

.. code-block:: python

   import torch
   import torch.nn as nn
   from dataclasses import dataclass
   from torch_timeseries.experiments import UEAClassificationExp

   class GRUClassifier(nn.Module):
       def __init__(self, n_features, n_classes, hidden=64):
           super().__init__()
           self.gru  = nn.GRU(n_features, hidden, batch_first=True)
           self.head = nn.Linear(hidden, n_classes)

       def forward(self, x):   # (B, T, C) → (B, n_classes)
           _, h = self.gru(x)
           return self.head(h.squeeze(0))

   @dataclass
   class MyClassExp(UEAClassificationExp):
       model_type: str = "GRUClassifier"

       def _init_model(self):
           self.model = GRUClassifier(
               self.dataset.num_features, self.dataset.num_classes
           ).to(self.device)

       def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
           x = batch_x.to(self.device, dtype=torch.float32)
           y = batch_y.to(self.device, dtype=torch.long).squeeze(-1)
           return self.model(x), y

   result = MyClassExp(
       dataset_type="EthanolConcentration", windows=1751,
       epochs=30, device="cuda",
   ).run(seed=1)
   print(result)   # {'accuracy': ...}

Irregular Time Series — Interpolation and Forecasting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Handle real-world clinical or sensor datasets where observations arrive at
irregular timestamps with per-feature missingness. Use
``IrregularInterpolationDataModule`` to reconstruct held-out query points, or
``IrregularForecastDataModule`` to predict future irregular observations.

.. code-block:: python

   import torch
   import torch.nn as nn
   from dataclasses import dataclass
   from torch_timeseries.dataset.irregular import PhysioNet2012
   from torch_timeseries.dataloader.v2 import (
       IrregularInterpolationDataModule, IrregularInterpolationConfig,
       SplitConfig, LoaderConfig,
   )
   from torch_timeseries.scaler import StandardScaler

   ds = PhysioNet2012(root="./data")

   dm = IrregularInterpolationDataModule(
       dataset=ds,
       scaler=StandardScaler(),
       window=IrregularInterpolationConfig(query_rate=0.2),
       split=SplitConfig(train=0.7, val=0.1, test=0.2),
       loader=LoaderConfig(batch_size=32),
   )

   # batch fields: x (B,T,F), t (B,T), mask (B,T,F),
   #               y (B,Tq,F), t_query (B,Tq), query_mask (B,Tq,F)
   batch = next(iter(dm.train_loader))

Use the built-in GRU-D model and experiment runner:

.. code-block:: python

   from torch_timeseries.experiments.GRUD import GRUDIrregularInterpolation

   result = GRUDIrregularInterpolation(
       dataset_type="PhysioNet2012", hidden_size=64,
       epochs=50, batch_size=64, device="cuda",
   ).run(seed=1)
   print(result)   # {'mse': ..., 'mae': ...}

For forecasting, swap in ``IrregularForecastDataModule``
(``obs_frac`` controls how much of each sequence is used as context):

.. code-block:: python

   from torch_timeseries.experiments.GRUD import GRUDIrregularForecast

   result = GRUDIrregularForecast(
       dataset_type="PhysioNet2012", obs_frac=0.7,
       hidden_size=64, epochs=50, device="cuda",
   ).run(seed=1)
   print(result)   # {'mse': ..., 'mae': ...}

Time Series Generation with Diffusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Synthesise realistic sequences with NsDiff (Non-Stationary DDPM). Works on
any dataset — here on a small custom tensor dataset.

.. code-block:: python

   import torch
   from torch_timeseries.model.NsDiff import NsDiff
   from torch_timeseries.experiments.NsDiff import NsDiffGeneration

   # ── Custom loop ───────────────────────────────────────────────────────────
   T, C = 96, 3
   real   = torch.randn(400, T, C)
   loader = torch.utils.data.DataLoader(
       torch.utils.data.TensorDataset(real), batch_size=64, shuffle=True
   )

   model = NsDiff(seq_len=T, n_features=C, T=100, kernel_size=24)
   opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

   for epoch in range(50):
       for (x,) in loader:
           opt.zero_grad()
           model.loss(x).backward()
           opt.step()

   samples = model.generate(n=16)   # (16, 96, 3)

   # ── Experiment runner on the built-in Sine benchmark ─────────────────────
   result = NsDiffGeneration(
       dataset_type="Sine", seq_len=24, T=50,
       epochs=300, device="cuda",
   ).run(seed=1)
   print(result)
   # {'discriminative_score': ..., 'predictive_score': ...,
   #  'context_fid': ...,        'correlational_score': ...}
