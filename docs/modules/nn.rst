torch_timeseries.nn
===================

.. contents:: Contents
    :local:

Reusable neural network building blocks.  All modules are importable from
``torch_timeseries.nn`` and can be composed freely into custom models.


Temporal Encoding
-----------------

Methods for encoding time position information into dense vectors.  These are
drop-in replacements for (or supplements to) the fixed sinusoidal encoding
used in standard Transformers.

.. currentmodule:: torch_timeseries.nn

.. autosummary::
    :nosignatures:
    :toctree: ../generated
    :template: autosummary/only_class.rst

    Time2Vec
    LearnableFourierFeatures
    RotaryEmbedding
    SinusoidalEmbedding


Data Embeddings
---------------

Composite embeddings that combine value projection, positional encoding, and
optional time-feature encoding.  Used internally by the forecasting models.

.. autosummary::
    :nosignatures:
    :toctree: ../generated
    :template: autosummary/only_class.rst

    DataEmbedding
    DataEmbedding_wo_pos
    DataEmbedding_inverted
    PatchEmbedding
    TokenEmbedding
    PositionalEmbedding
    TemporalEmbedding
    TimeFeatureEmbedding
