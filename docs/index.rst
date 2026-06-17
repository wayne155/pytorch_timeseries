
.. raw:: html

   <div class="hero">
     <div class="hero-title">torch-timeseries</div>
     <p class="hero-subtitle">
       A research toolkit covering <strong>six time-series tasks</strong> — forecasting,
       probabilistic forecasting, generation, imputation, anomaly detection, and
       classification — with 20 built-in models, ready-made datasets, and a two-line
       experiment runner.
     </p>
     <div class="hero-badges">
       <a href="https://pypi.org/project/torch-timeseries/"><img src="https://img.shields.io/pypi/v/torch-timeseries?color=2563eb&style=flat-square" alt="PyPI"/></a>
       <a href="https://pytorch-timeseries.readthedocs.io/"><img src="https://img.shields.io/readthedocs/pytorch-timeseries?style=flat-square" alt="Docs"/></a>
       <a href="https://github.com/wayne155/pytorch_timeseries"><img src="https://img.shields.io/github/stars/wayne155/pytorch_timeseries?style=flat-square" alt="Stars"/></a>
       <img src="https://img.shields.io/pypi/pyversions/torch-timeseries?style=flat-square" alt="Python"/>
     </div>
     <div class="hero-install">pip install torch-timeseries</div>
   </div>


----

.. raw:: html

   <p class="section-label">Time-Series Tasks</p>

.. grid:: 2 2 3 3
   :gutter: 3

   .. grid-item-card:: Forecasting
      :img-top: _static/img/forecast_custom_pipeline.png
      :link: get-started/quickstart
      :link-type: doc

      Predict future values from historical context. Use `ForecastDataModule`
      with any model — linear, attention-based, or MLP-mixer.

   .. grid-item-card:: Probabilistic Forecasting
      :img-top: _static/img/prob_forecast.png
      :link: concepts/experiments
      :link-type: doc

      Quantify uncertainty with calibrated prediction intervals. `ProbForecastExp`
      reports CRPS, QICE, PICP and more.

   .. grid-item-card:: Generation
      :img-top: _static/img/nsdiff_generation.png
      :link: concepts/experiments
      :link-type: doc

      Synthesise realistic time-series via diffusion. Six models including
      NsDiff, TMDM, TimeGAN, CSDI and DiffusionTS.

   .. grid-item-card:: Imputation
      :img-top: _static/img/imputation.png
      :link: concepts/experiments
      :link-type: doc

      Reconstruct missing values under random or block masking.
      `ImputationExp` handles masking and metric computation automatically.

   .. grid-item-card:: Anomaly Detection
      :img-top: _static/img/anomaly_detection.png
      :link: concepts/experiments
      :link-type: doc

      Detect anomalies from reconstruction error. `AnomalyDetectionExp`
      reports precision, recall and F1 against ground-truth labels.

   .. grid-item-card:: Classification
      :img-top: _static/img/classification.png
      :link: concepts/experiments
      :link-type: doc

      Label multivariate time-series from the UEA archive.
      `UEAClassificationExp` wraps sktime datasets and reports accuracy.


----

.. raw:: html

   <p class="section-label">Key features</p>

.. grid:: 2 2 4 4
   :gutter: 2

   .. grid-item-card:: 20 built-in models
      :text-align: center

      .. raw:: html

         <div class="feature-icon">🏗️</div>

      DLinear · PatchTST · iTransformer · Autoformer · FEDformer · NsDiff · TMDM
      and 13 more, ready to benchmark.

   .. grid-item-card:: Two usage modes
      :text-align: center

      .. raw:: html

         <div class="feature-icon">⚡</div>

      **Way 1** — drop-in DataModules for custom loops.
      **Way 2** — one-line `Experiment` runner with result persistence.

   .. grid-item-card:: Temporal encodings
      :text-align: center

      .. raw:: html

         <div class="feature-icon">📐</div>

      `Time2Vec`, `LearnableFourierFeatures`, `RotaryEmbedding` (RoPE) and
      `SinusoidalEmbedding` in `torch_timeseries.nn`.

   .. grid-item-card:: Leaderboards & grids
      :text-align: center

      .. raw:: html

         <div class="feature-icon">📊</div>

      Grid-search over models × datasets × seeds. Save, compare and render
      leaderboard tables automatically.


----

.. raw:: html

   <p class="section-label">Quick example</p>

.. tab-set::

   .. tab-item:: Experiment runner

      .. code-block:: python

         from torch_timeseries import Experiment

         results = Experiment(
             model="PatchTST",
             task="Forecast",
             dataset="ETTh1",
             windows=96,
             pred_len=96,
         ).run(seeds=[1, 2, 3])

         print(results[0].metrics)   # MAE, MSE, RMSE, MAPE, MSPE

   .. tab-item:: Custom pipeline

      .. code-block:: python

         from torch_timeseries.dataset import ETTh1
         from torch_timeseries.dataloader.v2 import ForecastDataModule, WindowConfig

         dm = ForecastDataModule(
             dataset=ETTh1("./data"),
             window=WindowConfig(window=96, horizon=1, steps=96),
         )

         for batch in dm.train_loader:
             x = batch.x.float()   # (B, 96, C)
             y = batch.y.float()   # (B, 96, C)

   .. tab-item:: CLI

      .. code-block:: bash

         # Run 3 seeds of PatchTST on ETTh1
         pytexp --model PatchTST --task Forecast --dataset_type ETTh1 runs '[1,2,3]'

         # Grid search
         # Edit a grid config then:
         pytexp compare --save_dir ./results --task Forecast


----

.. toctree::
   :maxdepth: 1
   :caption: Installation & Getting Started
   :hidden:

   install/install
   get-started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: Concepts
   :hidden:
   :titlesonly:

   concepts/experiments
   concepts/results-and-artifacts
   concepts/dataloader
   concepts/architecture
   leaderboard/index
   changelog

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:
   :titlesonly:

   modules/dataset
   modules/dataloader
   modules/scaler
   modules/model
   modules/nn

.. toctree::
   :maxdepth: 1
   :caption: Contribute
   :hidden:
   :titlesonly:

   contribute/quick-contribute
