torch_timeseries.metrics
========================

Evaluation metrics for time series tasks.  All classes are
:class:`torchmetrics.Metric` subclasses and are compatible with
:class:`torchmetrics.MetricCollection`.

----

Point Forecast Metrics
----------------------

These metrics evaluate deterministic (point) forecasts and complement the
standard MSE / MAE / RMSE provided by ``torchmetrics``.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Class / Function
     - Description
   * - ``SMAPE``
     - Symmetric Mean Absolute Percentage Error ∈ [0, 200].  Scale-free
       and symmetric — suitable for comparing models across series of
       different magnitudes.
   * - ``MASE``
     - Mean Absolute Scaled Error.  Scales the MAE by the in-sample naive
       forecast error (call ``set_naive_mae()`` or pass ``naive_mae`` to
       the constructor).
   * - ``QuantileLoss``
     - Pinball loss at a specified quantile level ``q ∈ (0, 1)``.  At
       ``q=0.5`` this equals half the MAE (equivalent to the median
       regression loss).
   * - ``naive_seasonal_mae``
     - Helper that computes the seasonal naive forecast MAE from a training
       series — use as the ``naive_mae`` argument to ``MASE``.

.. code-block:: python

   from torchmetrics import MetricCollection
   from torch_timeseries.metrics import SMAPE, MASE, QuantileLoss, naive_seasonal_mae

   # Compute naive MAE from training data
   naive_mae = naive_seasonal_mae(train_series, seasonality=24)

   metrics = MetricCollection({
       "smape": SMAPE(),
       "mase": MASE(naive_mae=naive_mae),
       "q50":  QuantileLoss(quantile=0.5),
       "q90":  QuantileLoss(quantile=0.9),
   })

   for pred, target in test_loader:
       metrics.update(pred, target)

   print(metrics.compute())

.. currentmodule:: torch_timeseries.metrics

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   SMAPE
   MASE
   QuantileLoss

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   naive_seasonal_mae

----

Probabilistic Forecast Metrics
-------------------------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Class
     - Description
   * - ``CRPS``
     - Continuous Ranked Probability Score — strictly proper scoring rule
       for distributional forecasts.  Accepts Monte-Carlo sample sets.
   * - ``CRPSSum``
     - CRPS summed over features (multivariate variant).
   * - ``PICP``
     - Prediction Interval Coverage Probability — fraction of targets
       falling inside a given interval.
   * - ``QICE``
     - Quantile Interval Coverage Error — signed deviation from the
       nominal coverage at a given quantile level.
   * - ``ProbMAE``
     - Mean absolute error evaluated on the predicted mean.
   * - ``ProbMSE``
     - Mean squared error evaluated on the predicted mean.
   * - ``ProbRMSE``
     - Root mean squared error evaluated on the predicted mean.

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   CRPS
   CRPSSum
   PICP
   QICE
   ProbMAE
   ProbMSE
   ProbRMSE

----

Generation Metrics
------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Function
     - Description
   * - ``discriminative_score``
     - Post-hoc binary classification accuracy for distinguishing real
       from synthetic series (lower = better synthesis).
   * - ``predictive_score``
     - Train-on-synthetic, test-on-real MAE (lower = better).
   * - ``context_fid``
     - Fréchet Inception Distance in the latent space of a trained
       context encoder (lower = better).
   * - ``correlational_score``
     - Absolute difference between feature-level auto-correlations of
       real and synthetic series.

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   discriminative_score
   predictive_score
   context_fid
   correlational_score
