Advanced Usage
==============

Fast Evaluation Windows
-----------------------

By default, both training and evaluation use a dense sliding window — the
window shifts by one step each time. For tasks where inference is expensive
(e.g. a diffusion model that samples 100 trajectories per window), this can
make evaluation prohibitively slow.

``WindowConfig.fast_val`` and ``fast_test`` switch the val/test split to
**non-overlapping** windows (stride = ``window + horizon + steps − 1``)
while training keeps the dense sliding window:

.. code-block:: python

   from torch_timeseries.dataset import ETTh1
   from torch_timeseries.scaler import StandardScaler
   from torch_timeseries.dataloader.v2 import ForecastDataModule, WindowConfig

   dm = ForecastDataModule(
       dataset=ETTh1(),
       scaler=StandardScaler(),
       window=WindowConfig(
           window=96, steps=24,
           fast_val=True,
           fast_test=True,
       ),
   )
   # ETTh1 pred_len 24:  val/test windows  2857 → 24  (119× fewer model calls)

The windows still tile the full evaluation span so metrics remain
representative — they are computed on disjoint windows rather than every
shifted copy.

.. image:: ../_static/img/fast_eval_windows.png
   :alt: Dense training windows (top) vs. non-overlapping fast eval windows (bottom)
   :align: center

Probabilistic Forecasting
--------------------------

Any model that can be called multiple times to produce different predictions
(MC Dropout, deep ensembles, diffusion) fits into ``ProbForecastExp``. The
engine aggregates ``(B, pred_len, C, n_samples)`` tensors and evaluates
CRPS, QICE, PICP, and calibration metrics automatically.

.. code-block:: python

   import torch
   import torch.nn as nn
   from dataclasses import dataclass
   from torch_timeseries.experiments import ProbForecastExp

   class MCDropoutForecaster(nn.Module):
       def __init__(self, seq_len, pred_len, drop=0.15):
           super().__init__()
           self.net = nn.Sequential(
               nn.Linear(seq_len, 256), nn.ReLU(), nn.Dropout(drop),
               nn.Linear(256, pred_len),
           )

       def forward(self, x):           # (B, T, C) → (B, pred_len, C)
           return self.net(x.transpose(1, 2)).transpose(1, 2)

       def sample(self, x, n=200):     # (B, pred_len, C, n)
           self.train()
           with torch.no_grad():
               return torch.stack([self(x) for _ in range(n)], dim=-1)

   @dataclass
   class MCDropoutExp(ProbForecastExp):
       model_type: str = "MCDropout"

       def _init_model(self):
           self.model = MCDropoutForecaster(
               self.windows, self.pred_len
           ).to(self.device)

       def _process_train_batch(self, batch):
           x = batch.x.float().to(self.device)
           y = batch.y.float().to(self.device)
           self.model.train()
           return nn.MSELoss()(self.model(x), y)

       def _process_val_batch(self, batch):
           x = batch.x.float().to(self.device)
           y = batch.y.float().to(self.device)
           preds = self.model.sample(x, n=self.num_samples)   # (B, O, C, S)
           return preds, y

   result = MCDropoutExp(
       dataset_type="ETTh1", windows=96, pred_len=24,
       num_samples=200, epochs=30, device="cuda",
   ).run(seed=0)
   # → {'crps': ..., 'picp': ..., 'qice': ..., 'prob_mse': ...}

Custom Model Registration
--------------------------

To use the default experiment runner with your own model, subclass the task
experiment and register it by name:

.. code-block:: python

   from dataclasses import dataclass
   import torch.nn as nn
   from torch_timeseries import Experiment, register_model
   from torch_timeseries.experiments import ForecastExp

   class MyNet(nn.Module):
       def __init__(self, seq_len, pred_len):
           super().__init__()
           self.proj = nn.Linear(seq_len, pred_len)

       def forward(self, x):   # (B, T, C) → (B, pred_len, C)
           return self.proj(x.transpose(1, 2)).transpose(1, 2)

   @dataclass
   class MyForecastExp(ForecastExp):
       model_type: str = "MyNet"

       def _init_model(self):
           self.model = MyNet(self.windows, self.pred_len).to(self.device)

   register_model(MyForecastExp)

   results = Experiment(
       model="MyNet", task="Forecast",
       dataset="ETTh1", windows=96, pred_len=96,
   ).run(seeds=[1, 2, 3])

After ``register_model``, the model is also available from the CLI:

.. code-block:: bash

   pytexp --model MyNet --task Forecast --dataset_type ETTh1 run 1

Grid Search
-----------

``Experiment.grid`` runs every combination of models × tasks × datasets × seeds
and writes results to disk:

.. code-block:: python

   from torch_timeseries import Experiment

   Experiment.grid(
       models=["DLinear", "PatchTST", "iTransformer"],
       tasks=["Forecast"],
       datasets=["ETTh1", "ETTm1", "Weather"],
       seeds=[1, 2, 3],
       windows=96,
       pred_len=96,
       save_dir="./results",
   ).run()

   # Compare afterwards
   Experiment.compare(save_dir="./results", task="Forecast")

CLI
---

Every experiment has a command-line equivalent. The module that calls
``register_model(...)`` must be importable on the Python path.

.. code-block:: bash

   # Forecasting — 3 seeds
   pytexp --model DLinear --task Forecast --dataset_type ETTh1 runs '[1,2,3]'

   # Imputation
   pytexp --model PatchTST --task Imputation --dataset_type ETTh1 run 1

   # Anomaly detection
   pytexp --model TimesNet --task AnomalyDetection --dataset_type MSL run 1

   # Classification
   pytexp --model DLinear --task UEAClassification \
          --dataset_type EthanolConcentration run 1

   # Compare saved results
   pytexp compare --save_dir ./results --task Forecast

Column Selection
----------------

Forecast windows can observe a subset of features and target a different subset:

.. code-block:: python

   from torch_timeseries.dataloader.v2 import WindowConfig

   WindowConfig(
       window=96,
       horizon=1,
       steps=96,
       input_columns=["HUFL", "HULL", "MUFL"],   # observed features
       target_columns=["OT"],                      # feature(s) to predict
   )

Useful for multivariate datasets where the model should read sensor
readings but predict only a single target variable.
