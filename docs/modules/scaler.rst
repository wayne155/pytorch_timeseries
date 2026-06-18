torch_timeseries.scaler
========================

Scalers normalise the time series before windowing. They are fitted on the
training split and applied consistently across train, val, and test.

Pass a scaler to any DataModule constructor:

.. code-block:: python

   from torch_timeseries.scaler import StandardScaler
   from torch_timeseries.dataloader.v2 import ForecastDataModule

   dm = ForecastDataModule(dataset=..., scaler=StandardScaler(), ...)

Batches always carry **scaled** values in ``batch.x`` and ``batch.y``.
Unscaled values are available in ``batch.x_raw`` and ``batch.y_raw`` when
``WindowConfig(include_raw=True)`` is set.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Scaler
     - Description
   * - ``StandardScaler``
     - Zero-mean, unit-variance normalisation (z-score). Default choice for
       most forecasting and imputation benchmarks.
   * - ``MaxAbsScaler``
     - Scales each feature to the range [−1, 1] by dividing by the maximum
       absolute value.  Preserves sparsity and zero entries.

----

.. currentmodule:: torch_timeseries.scaler

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_timeseries.scaler.scalers %}
     {{ name }}
   {% endfor %}
