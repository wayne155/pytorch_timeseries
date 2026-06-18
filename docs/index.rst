
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

   <div class="section-divider"></div>
   <p class="section-label">Time-Series Tasks</p>


.. grid:: 2 2 3 3
   :gutter: 3

   .. grid-item-card:: Forecasting
      :img-top: _static/img/forecast_custom_pipeline.png
      :link: get-started/quickstart
      :link-type: doc

      Predict future values from historical context. Use ``ForecastDataModule``
      with any model — linear, attention-based, or MLP-mixer.

   .. grid-item-card:: Probabilistic Forecasting
      :img-top: _static/img/prob_forecast.png
      :link: concepts/experiments
      :link-type: doc

      Quantify uncertainty with calibrated prediction intervals. ``ProbForecastExp``
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
      ``ImputationExp`` handles masking and metric computation automatically.

   .. grid-item-card:: Anomaly Detection
      :img-top: _static/img/anomaly_detection.png
      :link: concepts/experiments
      :link-type: doc

      Detect anomalies from reconstruction error. ``AnomalyDetectionExp``
      reports precision, recall and F1 against ground-truth labels.

   .. grid-item-card:: Classification
      :img-top: _static/img/classification.png
      :link: concepts/experiments
      :link-type: doc

      Label multivariate time-series from the UEA archive.
      ``UEAClassificationExp`` wraps sktime datasets and reports accuracy.


.. raw:: html

   <div class="section-divider"></div>
   <p class="section-label">20 Built-in Models</p>

   <div class="model-showcase">

     <div class="model-group">
       <div class="model-group-header forecast-header">
         <span class="model-group-icon">📈</span>
         <span class="model-group-title">Forecasting</span>
         <span class="model-group-count">14 models</span>
       </div>
       <div class="model-pills">
         <a href="generated/torch_timeseries.model.DLinear.html" class="mpill mpill-forecast">DLinear</a>
         <a href="generated/torch_timeseries.model.NLinear.html" class="mpill mpill-forecast">NLinear</a>
         <a href="generated/torch_timeseries.model.PatchTST.html" class="mpill mpill-forecast">PatchTST</a>
         <a href="generated/torch_timeseries.model.iTransformer.html" class="mpill mpill-forecast">iTransformer</a>
         <a href="generated/torch_timeseries.model.Informer.html" class="mpill mpill-forecast">Informer</a>
         <a href="generated/torch_timeseries.model.Autoformer.html" class="mpill mpill-forecast">Autoformer</a>
         <a href="generated/torch_timeseries.model.FEDformer.html" class="mpill mpill-forecast">FEDformer</a>
         <a href="generated/torch_timeseries.model.TSMixer.html" class="mpill mpill-forecast">TSMixer</a>
         <a href="generated/torch_timeseries.model.Crossformer.html" class="mpill mpill-forecast">Crossformer</a>
         <a href="generated/torch_timeseries.model.SCINet.html" class="mpill mpill-forecast">SCINet</a>
         <a href="generated/torch_timeseries.model.TimesNet.html" class="mpill mpill-forecast">TimesNet</a>
         <a href="generated/torch_timeseries.model.CATS.html" class="mpill mpill-forecast">CATS</a>
         <a href="generated/torch_timeseries.model.FITS.html" class="mpill mpill-forecast">FITS</a>
         <a href="generated/torch_timeseries.model.FreTS.html" class="mpill mpill-forecast">FreTS</a>
       </div>
       <p class="model-group-note">Most models also support imputation, anomaly detection &amp; classification.</p>
     </div>

     <div class="model-group">
       <div class="model-group-header gen-header">
         <span class="model-group-icon">✨</span>
         <span class="model-group-title">Generation</span>
         <span class="model-group-count">6 models</span>
       </div>
       <div class="model-pills">
         <a href="generated/torch_timeseries.model.TimeGAN.html" class="mpill mpill-gen">TimeGAN</a>
         <a href="generated/torch_timeseries.model.CSDI.html" class="mpill mpill-gen">CSDI</a>
         <a href="generated/torch_timeseries.model.DiffusionTS.html" class="mpill mpill-gen">DiffusionTS</a>
         <a href="generated/torch_timeseries.model.TimeDiff.html" class="mpill mpill-gen">TimeDiff</a>
         <a href="generated/torch_timeseries.model.NsDiff.html" class="mpill mpill-gen">NsDiff</a>
         <a href="generated/torch_timeseries.model.TMDM.html" class="mpill mpill-gen">TMDM</a>
       </div>
       <p class="model-group-note">GAN · Score-based diffusion · Non-stationary DDPM</p>
     </div>

   </div>

   <div class="section-divider"></div>
   <p class="section-label">Key Features</p>


.. grid:: 2 2 3 3
   :gutter: 3

   .. grid-item-card:: Two usage modes
      :text-align: center

      .. raw:: html

         <div class="feature-icon">⚡</div>

      **Way 1** — drop-in DataModules for your own training loop.
      **Way 2** — one-line ``Experiment`` runner with automatic result persistence.

   .. grid-item-card:: Temporal encodings
      :text-align: center

      .. raw:: html

         <div class="feature-icon">📐</div>

      ``Time2Vec``, ``LearnableFourierFeatures``, ``RotaryEmbedding`` (RoPE) and
      ``SinusoidalEmbedding`` — all in ``torch_timeseries.nn``.

   .. grid-item-card:: Leaderboards & grids
      :text-align: center

      .. raw:: html

         <div class="feature-icon">📊</div>

      Grid-search over models × datasets × seeds. Save, compare and render
      leaderboard tables automatically.


.. raw:: html

   <div class="section-divider"></div>
   <p class="section-label">Quick Start</p>


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

         # Compare saved results
         pytexp compare --save_dir ./results --task Forecast


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
