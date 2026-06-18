torch_timeseries.nn
===================

Reusable neural network building blocks. All modules are importable from
``torch_timeseries.nn`` and can be composed freely into custom models.

----

Temporal Encoding
-----------------

Methods for encoding scalar time positions into dense vectors.  Use these as
drop-in replacements for (or complements to) the fixed sinusoidal encoding used
in standard Transformers.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Summary
   * - ``Time2Vec``
     - 1 linear trend component + *k−1* learnable-frequency sinusoids.
       Input ``(B, L)`` → output ``(B, L, k)``.
   * - ``LearnableFourierFeatures``
     - sin/cos pairs with learned frequencies; fully differentiable.
       Input ``(B, L)`` → output ``(B, L, d_model)``.
   * - ``RotaryEmbedding``
     - RoPE — rotates query/key pairs so dot-products encode relative position.
       Operates on ``(B, H, L, dim)`` tensors; no extra parameters added.
   * - ``SinusoidalEmbedding``
     - Fixed geometric-frequency sin/cos table (no parameters). Classic
       Transformer positional encoding. Input ``(B, L, d)`` → output
       ``(1, L, d_model)``.

.. code-block:: python

   import torch
   from torch_timeseries.nn import Time2Vec, RotaryEmbedding

   # Time2Vec: encode timestep indices
   t2v = Time2Vec(k=16)
   tau = torch.arange(96).unsqueeze(0).float()   # (1, 96)
   enc = t2v(tau)                                  # (1, 96, 16)

   # RoPE: rotate query/key in an attention layer
   rope = RotaryEmbedding(dim=64)
   q = torch.randn(2, 8, 96, 64)   # (B, H, L, dim)
   k = torch.randn(2, 8, 96, 64)
   q_r, k_r = rope(q, k)

.. currentmodule:: torch_timeseries.nn

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   Time2Vec
   LearnableFourierFeatures
   RotaryEmbedding
   SinusoidalEmbedding

----

Data Embeddings
---------------

Composite embeddings that project raw input values into a ``d_model``-dimensional
space, optionally fusing positional encoding and time-feature encoding.
Used internally by the forecasting models.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Summary
   * - ``DataEmbedding``
     - Token + positional + temporal embedding. Standard choice for
       Informer / Autoformer / FEDformer.
   * - ``DataEmbedding_wo_pos``
     - Token + temporal embedding without positional encoding.
   * - ``DataEmbedding_inverted``
     - Inverted variant used by iTransformer — embeds features as tokens.
   * - ``PatchEmbedding``
     - Projects non-overlapping patches (used by PatchTST).
   * - ``TokenEmbedding``
     - 1-D causal convolution projecting each timestep to ``d_model``.
   * - ``PositionalEmbedding``
     - Fixed sinusoidal positional encoding table.
   * - ``TemporalEmbedding``
     - Learned calendar embeddings for month, day, weekday, hour, minute.
   * - ``TimeFeatureEmbedding``
     - Linear projection of engineered time features (e.g. from
       ``torch_timeseries.utils.timefeatures``).

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

----

Attention & Correlation
-----------------------

Attention mechanisms and correlation operators used by the built-in models.
These are available as standalone modules for custom architectures.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Summary
   * - ``FullAttention``
     - Standard scaled dot-product attention.
   * - ``ProbAttention``
     - ProbSparse attention from Informer — *O(L log L)* complexity.
   * - ``AttentionLayer``
     - Multi-head wrapper: projects to Q/K/V, calls inner attention, projects out.
   * - ``AutoCorrelation``
     - Period-based dependency discovery via FFT (Autoformer).
   * - ``AutoCorrelationLayer``
     - Multi-head wrapper for ``AutoCorrelation``.
   * - ``FourierBlock``
     - Fourier-domain mixing block (FEDformer).
   * - ``FourierCrossAttention``
     - Cross-attention in the Fourier domain.
   * - ``MultiWaveletTransform``
     - Multi-wavelet self-attention (FEDformer Wavelet variant).
   * - ``MultiWaveletCross``
     - Multi-wavelet cross-attention.

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   FullAttention
   ProbAttention
   AttentionLayer
   AutoCorrelation
   AutoCorrelationLayer
   FourierBlock
   FourierCrossAttention
   MultiWaveletTransform
   MultiWaveletCross

----

Encoder & Decoder
-----------------

Modular encoder/decoder stacks used internally by the Transformer-family models.

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   Encoder
   EncoderLayer
   EncoderStack
   ConvLayer
   Decoder
   DecoderLayer

----

Decomposition
-------------

Signal decomposition layers used by Autoformer, FEDformer, DLinear and others.

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   MovingAvg
   SeriesDecomp
   SeriesDecompMulti
