Forecaster — High-Level API
===========================

``Forecaster`` is a scikit-learn-style wrapper around any of the 86+ built-in
models. It handles training, evaluation, uncertainty quantification,
explainability, and deployment without boilerplate.

.. code-block:: python

   import numpy as np
   from torch_timeseries import Forecaster

   rng = np.random.default_rng(0)
   X   = rng.standard_normal((2_000, 7)).astype("float32")

   fc = Forecaster(
       "iTransformer",   # any registered model name
       seq_len  = 96,    # look-back window
       pred_len = 24,    # forecast horizon
       epochs   = 20,
       lr       = 1e-3,
       patience = 5,     # early stopping
       normalize= True,
   )

   fc.fit(X)             # train/val split, early stopping — all automatic

   ctx  = X[-96:]        # context window (96, 7)
   pred = fc.predict(ctx)       # → (24, 7)
   print(fc.score(X))           # {'MSE': 0.98, 'MAE': 0.79, 'RMSE': 0.99, …}

.. figure:: ../_static/img/forecaster_hero.png
   :alt: Forecaster context window and 24-step forecast with prediction intervals
   :align: center

   Context window (grey) · ground truth (blue dashed) · forecast (red) · 95% PI (shaded).

----

Fit · Predict · Score
---------------------

.. code-block:: python

   # Labeled DataFrame output
   df = fc.forecast_dataframe(
       ctx,
       channel_names=["OT","HUFL","HULL","MUFL","MULL","LUFL","LULL"],
       start_index=2_000,
   )

   # Scalar metrics
   print(fc.score(X))                      # {'MSE', 'MAE', 'RMSE', 'SMAPE'}
   print(fc.score_per_channel(X)["MAE"])   # (C,) per-channel MAE

   # Per-horizon metrics
   print(fc.summary_table(X))             # DataFrame: metric × step

   # Semantic error analysis
   print(fc.multistep_score(X, metric="mse"))   # {step: mse_value}
   print(fc.forecast_bias(X, channel=0))         # (pred_len,) signed error per step

----

Model Comparison & Leaderboard
-------------------------------

.. code-block:: python

   from torch_timeseries import compare, compare_plot

   X_train, X_test = X[:1_500], X[1_500:]

   results = compare(
       models  = ["DLinear", "NLinear", "PatchTST", "iTransformer", "TimeMixer"],
       X_train = X_train, X_test = X_test,
       seq_len = 96, pred_len = 24, epochs = 10,
   )

   # Sorted DataFrame leaderboard
   lb = Forecaster.leaderboard(
       X_train, X_test,
       ["DLinear", "NLinear", "PatchTST", "iTransformer", "TimeMixer"],
       seq_len=96, pred_len=24, epochs=10,
   )
   print(lb)

   # Auto-select the best model
   best = Forecaster.auto_select(X_train, X_test,
       candidates=["DLinear","PatchTST","iTransformer"], metric="mse")

.. figure:: ../_static/img/forecaster_comparison.png
   :alt: Model comparison bar chart
   :align: center

   Model comparison — ETTh1, seq=96, pred=24.

----

Uncertainty Quantification
--------------------------

Three complementary approaches — all return ``(pred_len, C)`` bounds:

.. code-block:: python

   # 1. MC-Dropout
   unc = fc.predict_uncertainty(ctx, n_samples=200)
   # {'mean': (24,7), 'std': (24,7), 'lower': (24,7), 'upper': (24,7)}

   # 2. Post-hoc calibration — calibrate interval width on held-out data
   fc.calibrate(X_train, target_coverage=0.90, n_samples=200)
   unc_cal = fc.predict_uncertainty(ctx, n_samples=200)

   # 3. Conformal prediction — distribution-free coverage guarantee
   intervals = fc.predict_interval(ctx, X_cal=X_train, coverage=0.90)
   # {'lower': (24,7), 'upper': (24,7)}

   # Reliability report (PICP, PINAW, CWC, Winkler)
   print(fc.reliability_score(X_train, X_test, coverage=0.90))

   # Fan chart
   fig = fc.plot_prediction_bands(X_test, coverages=(0.5, 0.8, 0.95))

.. figure:: ../_static/img/forecaster_uncertainty.png
   :alt: Forecast uncertainty fan chart
   :align: center

   Conformal prediction intervals at 50 / 80 / 95% coverage.

----

Signal Analysis & Diagnostics
------------------------------

All analysis methods are **static** — no model fitting required.

.. code-block:: python

   # Autocorrelation / PACF
   lags, acf  = Forecaster.autocorrelation(X, max_lag=48, channel=0)
   lags, pacf = Forecaster.partial_autocorrelation(X, max_lag=48, channel=0)

   # Power spectral density & spectrogram
   freqs, psd = Forecaster.spectral_density(X, channel=0)
   spec = Forecaster.spectrogram(X, channel=0, nperseg=64)    # STFT
   se   = Forecaster.spectral_entropy(X, channel=0)           # 0=pure tone, 1=noise

   # Seasonal decomposition
   decomp   = Forecaster.seasonal_decompose(X, period=24)
   strength = Forecaster.seasonal_strength(X, period=24)

   # Regime & change-point detection
   regimes = Forecaster.regime_detection(X, n_regimes=3, window=30)
   cps     = Forecaster.detect_change_points(X, window=40)

   # Preprocessing utilities
   trend, cycle = Forecaster.hodrick_prescott_filter(X, lam=1600)
   smoothed     = Forecaster.exponential_smoothing(X, alpha=0.3)
   X_normed, mu, sigma = Forecaster.z_normalize(X)
   x_bc, lam, offset   = Forecaster.box_cox_transform(X)

   # Residual diagnostics (fitted model required)
   lb  = fc.ljung_box(X_test, max_lag=20)         # Ljung-Box test
   diag = fc.forecast_diagnostic(X_test)           # comprehensive report

.. figure:: ../_static/img/forecaster_signal_analysis.png
   :alt: Signal analysis dashboard
   :align: center

   Raw series · ACF · Power spectral density · Trend extraction.

.. figure:: ../_static/img/forecaster_acf_pacf.png
   :alt: ACF and PACF side by side
   :align: center

   ACF and PACF of an AR(2) process — PACF cuts off after lag 2.

----

Explainability
--------------

.. code-block:: python

   # Input gradient saliency — which context timesteps drive the forecast?
   grad = fc.input_gradient(ctx, target_step=0, target_channel=0, absolute=True)
   # (seq_len, C) — large value = high influence

   fig = fc.plot_saliency(ctx, target_step=0, target_channel=0)

   # Global importance averaged over many windows
   global_imp = fc.explain_global(X_test, n_samples=100)

   # Permutation importance per channel
   ranking = fc.feature_importance_ranking(X_test, n_permutations=10)
   # [(ch_idx, importance), …] sorted by descending impact

.. figure:: ../_static/img/forecaster_saliency.png
   :alt: Input gradient saliency heatmap
   :align: center

   Input gradient saliency — context timesteps × channels.  Brighter = higher influence.

----

Transfer Learning & Serialisation
----------------------------------

.. code-block:: python

   # Pretrain, then fine-tune on a new domain
   fc_source = Forecaster("iTransformer", seq_len=96, pred_len=24, epochs=30)
   fc_source.fit(X_train)

   fc_target = fc_source.clone()                      # unfitted copy
   fc_target.copy_weights_from(fc_source)             # transplant weights
   fc_target.freeze_layers(["embedding"])             # lock some layers
   fc_target.partial_fit(X_new[:500], epochs=5)

   # Inspect memory
   print(fc_target.memory_usage())   # {total_params, trainable_params, size_mb}

   # Save / reload
   fc_source.save("./checkpoints/model")
   fc_reload = Forecaster.from_pretrained("./checkpoints/model", device="cpu")

   # Export to ONNX
   fc_source.to_onnx("./model.onnx")

----

Ensemble & Composition
-----------------------

.. code-block:: python

   from torch_timeseries import EnsembleForecaster, Pipeline, SklearnForecaster

   # Weighted ensemble of heterogeneous models
   ens = EnsembleForecaster(
       forecasters=[
           ("dlinear", Forecaster("DLinear",      seq_len=96, pred_len=24, epochs=10)),
           ("patchtst",Forecaster("PatchTST",     seq_len=96, pred_len=24, epochs=10)),
           ("itrans",  Forecaster("iTransformer", seq_len=96, pred_len=24, epochs=10)),
       ],
       weights=[0.2, 0.4, 0.4],
   )
   ens.fit(X_train)
   pred = ens.predict(X_test[:96])   # (24, 7)

   # sklearn-compatible wrapper (GridSearchCV / Pipeline ready)
   sk = SklearnForecaster("DLinear", seq_len=96, pred_len=24, epochs=5)
   sk.fit_ts(X_train)

----

Deployment Utilities
--------------------

.. code-block:: python

   # Latency / throughput profiling
   stats = fc.profile(X_test[:96], n_repeats=100)
   print(f"{stats['mean_ms']:.1f} ms  ·  {stats['throughput']:.0f} windows/s")

   # Memory-efficient rolling prediction on long series
   preds = fc.chunked_predict(X_test, chunk_size=64)   # (n_windows, pred_len, C)

   # Real-time streaming (one step at a time)
   for t, pred_step in fc.rolling_predict_iter(X_test, step=1):
       pass   # pred_step: (pred_len, C)

   # Export predictions to CSV
   fc.export_predictions(X_test, path="forecasts.csv")

   # Wrap as PyTorch Dataset for custom training loops
   ds     = fc.to_torch_dataset(X_train)
   loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

----

Full Method Reference
---------------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Method
     - Description
   * - ``fit(X)``
     - Train on time series array ``(T, C)``
   * - ``predict(ctx)``
     - Forecast next ``pred_len`` steps from context ``(seq_len, C)``
   * - ``score(X)``
     - Return MSE / MAE / RMSE / SMAPE dict
   * - ``score_per_channel(X)``
     - Per-channel metric dict
   * - ``summary_table(X)``
     - DataFrame of metrics at multiple horizons
   * - ``forecast_dataframe(ctx)``
     - Labeled DataFrame output
   * - ``predict_uncertainty(ctx, n_samples)``
     - MC-Dropout mean/std/lower/upper dict
   * - ``predict_interval(ctx, X_cal, coverage)``
     - Conformal prediction lower/upper dict
     -
   * - ``calibrate(X_train, target_coverage)``
     - Post-hoc calibration of interval width
   * - ``conformal_coverage(X_calib, X_test)``
     - Empirical coverage check dict
   * - ``winkler_score(X_calib, X_test)``
     - Winkler interval score (float)
   * - ``reliability_score(X_calib, X_test)``
     - PICP / PINAW / CWC / Winkler dict
   * - ``plot_prediction_bands(X)``
     - Fan chart with nested coverage bands
   * - ``leaderboard(X_train, X_test, models)`` *(classmethod)*
     - Ranked DataFrame of model metrics
   * - ``auto_select(X_train, X_val, candidates)`` *(classmethod)*
     - Fit all candidates, return best Forecaster
   * - ``compare_models(models, X_train, X_test)``
     - Compare against a list of model names
   * - ``score_vs_persistence(X_test)``
     - Model vs. naive persistence baseline
   * - ``autocorrelation(X)`` *(static)*
     - ACF values and lags
   * - ``partial_autocorrelation(X)`` *(static)*
     - PACF via Levinson-Durbin
   * - ``spectral_density(X)`` *(static)*
     - Power spectral density
   * - ``spectrogram(X)`` *(static)*
     - STFT spectrogram
   * - ``spectral_entropy(X)`` *(static)*
     - Shannon entropy of PSD
   * - ``seasonal_decompose(X, period)`` *(static)*
     - Trend / seasonal / residual decomposition
   * - ``seasonal_strength(X, period)`` *(static)*
     - Seasonal strength index [0, 1]
   * - ``trend_strength(X)`` *(static)*
     - Trend strength index [0, 1]
   * - ``cross_correlation(X)`` *(static)*
     - Cross-correlation function between channels
   * - ``granger_test(X, max_lag)`` *(static)*
     - Pairwise Granger causality F-statistics
   * - ``detect_change_points(X)`` *(static)*
     - Sliding-window change point detection
   * - ``regime_detection(X, n_regimes)`` *(static)*
     - k-means clustering on rolling statistics
   * - ``hodrick_prescott_filter(X)`` *(static)*
     - HP trend / cycle decomposition
   * - ``exponential_smoothing(X, alpha)`` *(static)*
     - SES or Holt's double exponential smoothing
   * - ``box_cox_transform(X)`` *(static)*
     - Box-Cox variance-stabilising transform
   * - ``z_normalize(X)`` *(static)*
     - Z-score normalise, return ``(X_normed, mu, std)``
   * - ``rolling_zscore(X, window)`` *(static)*
     - Sliding z-score anomaly indicator
   * - ``wavelet_decomposition(X)`` *(static)*
     - Haar DWT multi-level decomposition
   * - ``ljung_box(X, max_lag)``
     - Ljung-Box portmanteau test
   * - ``residual_acf(X)``
     - ACF of rolling forecast residuals
   * - ``forecast_diagnostic(X)``
     - One-shot residual / accuracy / bias report
   * - ``forecast_bias(X)``
     - Per-step mean signed error
   * - ``multistep_score(X, metric)``
     - Per-horizon metric dictionary
   * - ``input_gradient(ctx)``
     - Gradient saliency map ``(seq_len, C)``
   * - ``explain_global(X, n_samples)``
     - Average saliency over many windows
   * - ``feature_importance_ranking(X)``
     - Permutation importance, sorted by channel
   * - ``sensitivity_analysis(X, channel)``
     - Perturbation sensitivity of one channel
   * - ``memory_usage()``
     - ``{total_params, trainable_params, size_mb}``
   * - ``clone()``
     - Unfitted copy with identical hyperparameters
   * - ``copy_weights_from(other)``
     - Transplant weights from another Forecaster
   * - ``freeze_layers(names)``
     - Lock named parameter groups
   * - ``save(path)``
     - Serialise to disk
   * - ``from_pretrained(path)`` *(classmethod)*
     - Load from disk
   * - ``to_onnx(path)``
     - Export to ONNX runtime format
   * - ``profile(ctx, n_repeats)``
     - Measure latency and throughput
   * - ``set_device(device)``
     - Move model between CPU / GPU at runtime
   * - ``chunked_predict(X, chunk_size)``
     - Memory-efficient rolling prediction
   * - ``rolling_predict_iter(X, step)``
     - Generator of ``(t, pred)`` tuples
   * - ``export_predictions(X, path)``
     - Write rolling forecasts to CSV
   * - ``to_torch_dataset(X)``
     - Wrap as ``WindowDataset`` for DataLoader
   * - ``train_val_test_split(X)`` *(static)*
     - Chronological split into 3 arrays
   * - ``temporal_cross_validation(X, n_splits)``
     - Expanding-window cross-validation
   * - ``hyperparameter_search(X_train, X_val, param_grid)``
     - Random search over hyperparameter grid
   * - ``predict_bootstrap(X_train, X_test, n_boot)``
     - Block-bootstrap confidence intervals
   * - ``predict_autoregressive(X, n_steps)``
     - Recursive multi-step autoregressive forecast
   * - ``noise_robustness(X_train, X_test)``
     - Score degradation under additive noise
   * - ``prediction_stability(X_train, X_test, n_seeds)``
     - Variance of predictions across random seeds
   * - ``batch_evaluate(X_list, names)``
     - Evaluate on multiple datasets, return DataFrame
   * - ``concept_drift_score(X_ref, X_test)`` *(static)*
     - Jensen-Shannon divergence drift score
   * - ``feature_drift(X_ref, X_test)`` *(static)*
     - Per-channel JS divergence array
   * - ``correlation_network(X, threshold)`` *(static)*
     - Correlation graph edges above threshold
   * - ``get_target_correlations(X)`` *(static)*
     - CCF between each channel and the target
   * - ``to_lagged_features(X, lags)`` *(static)*
     - Convert to ``(features, targets)`` for sklearn
   * - ``seasonal_naive_baseline(X, period)`` *(static)*
     - Seasonal naïve forecast ``(pred_len, C)``
   * - ``functional_boxplot(X, period)`` *(static)*
     - Median ± IQR ribbon over complete cycles
   * - ``lag_plot(X, lag)`` *(static)*
     - Scatter ``x[t]`` vs ``x[t-lag]``
   * - ``seasonal_plot(X, period)`` *(static)*
     - Overlaid seasonal lines
   * - ``histogram_forecast(X)``
     - Overlaid actual vs predicted histogram
   * - ``compare_channel_forecasts(X)``
     - Multi-panel per-channel forecast grid
   * - ``plot_learning_curve(X)``
     - Learning curve as training size grows
   * - ``forecast_with_trend(X, degree)``
     - Detrend → predict → re-add trend
   * - ``compute_pinball_loss(X, quantiles)``
     - Quantile / pinball loss at given quantiles
