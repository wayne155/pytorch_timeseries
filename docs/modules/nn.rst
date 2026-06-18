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

----

Normalization
-------------

Instance normalization utilities for distribution-shift robustness.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Class
     - Description
   * - ``RevIN``
     - Reversible Instance Normalization (Kim et al., ICLR 2022).
       Normalises the look-back window per instance and channel, runs the
       model on normalised inputs, then reverses the transformation on the
       forecast output.  Learnable affine parameters (γ, β) are applied
       after normalization.

.. code-block:: python

   from torch_timeseries.nn import RevIN

   revin = RevIN(num_features=7)
   x_norm = revin(x, mode="norm")      # x: (B, T, C)
   pred_norm = model(x_norm)           # (B, H, C)
   pred = revin(pred_norm, mode="denorm")

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   RevIN

----

Temporal Convolutional Networks
--------------------------------

Causal dilated convolutional building blocks for sequence modelling. TCNs
achieve the same sequence-to-sequence mapping as RNNs but are fully
parallelisable over the time axis during training.

Reference: Bai et al., *An Empirical Evaluation of Generic Convolutional and
Recurrent Networks for Sequence Modeling*, 2018.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Description
   * - ``CausalConv1d``
     - 1-D dilated causal convolution.  Left-pads so the output length always
       equals the input length.  The output at time ``t`` depends only on
       inputs ``≤ t``.
   * - ``TemporalBlock``
     - One TCN residual block: two :class:`CausalConv1d` layers (ReLU +
       Dropout) with a residual skip connection.  A 1×1 projection is added
       automatically when input and output channels differ.
   * - ``TemporalConvNet``
     - Full TCN: a stack of :class:`TemporalBlock` layers with exponentially
       increasing dilation (``2^0, 2^1, …``).  The receptive field grows as
       ``1 + 2 × (k-1) × (2^n - 1)`` where ``k`` is the kernel size and
       ``n`` is the number of levels.

.. code-block:: python

   from torch_timeseries.nn import TemporalConvNet

   # Input: (B, C, L) — channels-first convention (same as Conv1d)
   tcn = TemporalConvNet(in_channels=7, num_channels=[64, 64, 64], kernel_size=3)

   x   = torch.randn(4, 7, 96)       # (B, in_channels, L)
   out = tcn(x)                       # (4, 64, 96)

   # For (B, L, C) inputs — transpose before and after
   x_blc = torch.randn(4, 96, 7)
   out_blc = tcn(x_blc.transpose(1, 2)).transpose(1, 2)  # (4, 96, 64)

.. currentmodule:: torch_timeseries.nn

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   CausalConv1d
   TemporalBlock
   TemporalConvNet

----

Patching
--------

Decompose a time series into fixed-length overlapping patches, following
PatchTST (Nie et al., ICLR 2023).

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Class
     - Description
   * - ``Patcher``
     - Divides a ``(B, L, C)`` sequence into ``(B, N, patch_len, C)`` patches.
       Supports non-overlapping (``stride=patch_len``), overlapping
       (``stride < patch_len``), and dense (``stride=1``) configurations.
       Three padding modes: ``'end'`` (replicate last step), ``'none'`` (no
       padding), or a constant integer.

.. code-block:: python

   from torch_timeseries.nn import Patcher

   patcher = Patcher(patch_len=16, stride=8)
   x   = torch.randn(4, 96, 7)   # (B, L, C)
   out = patcher(x)               # (4, 12, 16, 7) — 12 patches

   # Use inside a Transformer (flatten patch × channel):
   B, N, P, C = out.shape
   tokens = out.reshape(B, N, P * C)   # (4, 12, 112)

.. currentmodule:: torch_timeseries.nn

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   Patcher

----

MLP Blocks
----------

Standalone MLP and Mixer building blocks used by various architectures.

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Class
     - Description
   * - ``FeedForward``
     - Two-layer MLP ``(linear → act → dropout → linear → dropout)``,
       the standard Transformer FFN.  Accepts any ``(*, d_model)`` input.
       Activation can be ``'relu'``, ``'gelu'``, ``'silu'``, ``'tanh'``, or
       an ``nn.Module``.
   * - ``MixerBlock``
     - Channel-mixing + time-mixing MLP block (TSMixer / MLP-Mixer style).
       Alternating 1-D MLPs on the time and channel axes with residual
       connections and layer normalization.

.. code-block:: python

   from torch_timeseries.nn import FeedForward, MixerBlock

   # Drop-in FFN for a Transformer encoder
   ffn = FeedForward(d_model=256, d_ff=1024, activation='gelu', dropout=0.1)
   x   = torch.randn(4, 96, 256)
   out = ffn(x)   # (4, 96, 256)

   # Standalone MLP-Mixer block
   block = MixerBlock(seq_len=96, d_model=64)
   out   = block(torch.randn(4, 96, 64))  # (4, 96, 64)

.. currentmodule:: torch_timeseries.nn

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   FeedForward
   MixerBlock
