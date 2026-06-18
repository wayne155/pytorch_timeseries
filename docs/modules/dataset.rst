torch_timeseries.dataset
========================

Datasets download and cache benchmark data automatically. Every dataset is a
subclass of ``TimeSeriesDataset`` and exposes three attributes after loading:

.. code-block:: python

   from torch_timeseries.dataset import ETTh1

   dataset = ETTh1()                 # downloads to ~/.torchtimeseries/data/
   print(dataset.data.shape)         # (T, num_features)  numpy array
   print(dataset.num_features)       # int
   print(dataset.dates.head())       # DataFrame with a 'date' column

Pass any dataset directly to a v2 DataModule — see :doc:`../concepts/dataloader`.

To load a local CSV, use :func:`build_dataset`:

.. code-block:: python

   from torch_timeseries.dataset import build_dataset

   dataset = build_dataset(csv="./sensors.csv", freq="h")

To create a reusable dataset class, see :doc:`../notes/create_dataset`.

----

Forecasting & Imputation Datasets
----------------------------------

Standard benchmark time series used for long- and short-term forecasting and
for imputation.  All have a datetime index and multiple numeric features.

.. currentmodule:: torch_timeseries.dataset

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_timeseries.dataset.forecast_datasets %}
     {{ name }}
   {% endfor %}

Classification Datasets
-----------------------

Multivariate time-series classification benchmarks from the
`UEA archive <https://www.timeseriesclassification.com/>`__.
The ``UEA`` dataset accepts any archive name and downloads it automatically.

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_timeseries.dataset.classify_datasets %}
     {{ name }}
   {% endfor %}

Anomaly Detection Datasets
--------------------------

Multivariate datasets with ground-truth anomaly labels, commonly used in
reconstruction-based anomaly detection research.

.. list-table::
   :widths: 15 85
   :header-rows: 0

   * - **SWaT**
     - Secure Water Treatment plant — 51 sensors, 11 days of attacks.
   * - **SMAP / MSL**
     - NASA Mars Science Laboratory and SMAP satellite telemetry.
   * - **SMD**
     - Server Machine Dataset — 28 machines, 38 metrics each.
   * - **PSM**
     - Pooled Server Metrics from Alibaba server data.

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_timeseries.dataset.anomaly_datasets %}
     {{ name }}
   {% endfor %}

Synthetic Datasets
------------------

Lightweight synthetic datasets for unit testing and quick prototyping.
They generate data on the fly — no download required.

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_timeseries.dataset.synthetic_datasets %}
     {{ name }}
   {% endfor %}
