torch_timeseries.dataset
==========================

.. contents:: Contents
    :local:



Forecast/Imputation Datasets
----------------------------

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

.. currentmodule:: torch_timeseries.dataset

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_timeseries.dataset.classify_datasets %}
     {{ name }}
   {% endfor %}



AnomalyDetection Datasets
-------------------------

.. currentmodule:: torch_timeseries.dataset

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_timeseries.dataset.anomaly_datasets %}
     {{ name }}
   {% endfor %}

Synthetic Datasets
----------------------

.. currentmodule:: torch_timeseries.dataset

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_timeseries.dataset.synthetic_datasets %}
     {{ name }}
   {% endfor %}
