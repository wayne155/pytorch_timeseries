torch_timeseries.model
======================

.. contents:: Contents
    :local:

This module contains all deep learning models supported by pytorch_timeseries.
Models are grouped by their primary task; many forecasting models also support
imputation, anomaly detection, and classification through the built-in
experiment runner.


Forecasting Models
------------------

These models are designed for supervised forecasting (and multi-task use via
the experiment runner).  All accept an input window of shape
``(batch, seq_len, n_features)`` and output predictions of shape
``(batch, pred_len, n_features)``.

.. currentmodule:: torch_timeseries.model

.. autosummary::
    :nosignatures:
    :toctree: ../generated
    :template: autosummary/only_class.rst

    {% for name in torch_timeseries.model.forecasting_models %}
      {{ name }}
    {% endfor %}


Generation Models
-----------------

These models learn to synthesise new time series windows unconditionally.
Each model exposes a ``loss(x)`` method for training and a
``generate(n, device)`` method that returns ``(n, seq_len, n_features)``
tensors of new sequences.

.. currentmodule:: torch_timeseries.model

.. autosummary::
    :nosignatures:
    :toctree: ../generated
    :template: autosummary/only_class.rst

    {% for name in torch_timeseries.model.generation_models %}
      {{ name }}
    {% endfor %}
