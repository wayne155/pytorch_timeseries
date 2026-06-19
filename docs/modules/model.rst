torch_timeseries.model
======================

**86 forecasting · 6 generation · 5 irregular** — 97 built-in models in total.
Click any card to view the full API — constructor arguments, paper reference, and task support.

.. list-table:: Task coverage at a glance — regular time series
   :widths: 22 12 12 12 12
   :header-rows: 1

   * - Model
     - Forecast
     - Imputation
     - Anomaly Det.
     - Classify
   * - DLinear / NLinear
     - ✓
     - ✓
     - ✓
     - ✓
   * - Informer / Autoformer / FEDformer
     - ✓
     - ✓
     - ✓
     - ✓
   * - PatchTST / iTransformer
     - ✓
     - ✓
     - ✓
     - ✓
   * - TSMixer / Crossformer / SCINet / TimesNet
     - ✓
     - ✓
     - ✓
     - ✓
   * - CATS / FITS / FreTS
     - ✓
     - —
     - —
     - —
   * - SegRNN / TimeMixer / TiDE / NHiTS
     - ✓
     - ✓
     - ✓
     - ✓
   * - TimeGAN / CSDI / DiffusionTS / TimeDiff / NsDiff / TMDM
     - —
     - —
     - —
     - — *(generation only)*

.. list-table:: Task coverage at a glance — irregular time series
   :widths: 22 14 14 14 14
   :header-rows: 1

   * - Model
     - Irregular Classify
     - Interpolation
     - Irregular Forecast
     - Notes
   * - GRU-D
     - ✓
     - ✓
     - ✓
     - no extra deps
   * - mTAN
     - ✓
     - ✓
     - ✓
     - no extra deps
   * - LatentODE
     - ✓
     - ✓
     - ✓
     - requires ``torchdiffeq``
   * - NeuralCDE
     - ✓
     - —
     - —
     - requires ``torchcde``
   * - Raindrop
     - ✓
     - —
     - —
     - requires ``torch_geometric``


----

Forecasting Models
------------------

All forecasting models accept ``(batch, seq_len, n_features)`` input and output
``(batch, pred_len, n_features)`` predictions.  Most also support imputation,
anomaly detection, and classification via the built-in experiment runner.
Use any model name with the :doc:`Forecaster API <../notes/forecaster>` for a
scikit-learn-style fit/predict/score interface.

**Transformer family**

.. list-table::
   :widths: 22 50 28
   :header-rows: 1

   * - Model
     - Key idea
     - Reference
   * - VanillaTransformer
     - Baseline encoder-decoder Transformer
     - `Vaswani et al., 2017 <https://arxiv.org/abs/1706.03762>`__
   * - Informer
     - ProbSparse attention *O(L log L)*, distilling encoder
     - `Zhou et al., AAAI 2021 <https://ojs.aaai.org/index.php/AAAI/article/view/17325>`__
   * - Autoformer
     - Auto-Correlation + progressive decomposition
     - `Wu et al., NeurIPS 2021 <https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html>`__
   * - FEDformer
     - Frequency-enhanced decomposed Transformer *O(L)*
     - `Zhou et al., ICML 2022 <https://proceedings.mlr.press/v162/zhou22g.html>`__
   * - NSTransformer
     - Non-stationary attention with de/stationarization
     - `Liu et al., NeurIPS 2022 <https://arxiv.org/abs/2205.14415>`__
   * - ETSformer
     - Exponential smoothing + Fourier cross-attention
     - `Woo et al., 2022 <https://arxiv.org/abs/2202.01381>`__
   * - PatchTST
     - Patch tokenisation, channel-independent Transformer
     - `Nie et al., ICLR 2023 <https://openreview.net/forum?id=Jbdc0vTOcol>`__
   * - Crossformer
     - Cross-time + cross-dimension two-stage attention
     - `Zhang & Yan, ICLR 2023 <https://openreview.net/forum?id=vSVLM2j9eie>`__
   * - iTransformer
     - Inverted attention — channels as tokens
     - `Liu et al., ICLR 2024 <https://openreview.net/forum?id=JePfAI8fah>`__
   * - Pathformer
     - Multi-scale patch Transformer with adaptive path routing
     - `Chen et al., ICLR 2024 <https://openreview.net/forum?id=lJkOCMP2aW>`__
   * - CATS
     - Auxiliary future queries extend context window
     - `Lin et al., ICML 2024 <https://arxiv.org/abs/2403.01673>`__
   * - Basisformer
     - Learnable seasonal-trend basis
     - —
   * - FiLM
     - Frequency-improved Legendre memory
     - `Zhou et al., NeurIPS 2022 <https://arxiv.org/abs/2205.08897>`__
   * - DiffTransformerForecaster
     - Differential attention reduces noise amplification
     - —
   * - FastFormerForecaster
     - Additive attention with global context
     - —
   * - LinearAttentionForecaster
     - Linear-complexity Transformer attention
     - —
   * - NystromForecaster
     - Nyström approximation of full attention
     - —
   * - SparseTransformerForecaster
     - Sparse local + global attention patterns
     - —
   * - AFTForecaster
     - Attention-free Transformer
     - —
   * - MEGAForecaster
     - Multi-head EMA + gated attention
     - —
   * - HyperForecaster
     - HyperNetwork-generated weights
     - —

**MLP / Linear family**

.. list-table::
   :widths: 22 50 28
   :header-rows: 1

   * - Model
     - Key idea
     - Reference
   * - DLinear
     - Decompose trend + seasonal, independent linear projections
     - `Zeng et al., AAAI 2023 <https://ojs.aaai.org/index.php/AAAI/article/view/26317>`__
   * - NLinear
     - Subtract last timestep before linear projection
     - `Zeng et al., AAAI 2023 <https://ojs.aaai.org/index.php/AAAI/article/view/26317>`__
   * - RLinear
     - Reversible normalisation + linear
     - —
   * - LightTS
     - Interval-enhanced dual-sampling MLP
     - `Zhang et al., 2022 <https://arxiv.org/abs/2207.01186>`__
   * - TSMixer
     - MLP-Mixer: alternating time-mix + feature-mix
     - `Chen et al., 2023 <https://arxiv.org/abs/2303.06053>`__
   * - TiDE
     - Time-series dense encoder/decoder (pure MLP)
     - `Das et al., TMLR 2023 <https://arxiv.org/abs/2304.08424>`__
   * - FreTS
     - All-MLP in the frequency domain
     - `Yi et al., NeurIPS 2023 <https://arxiv.org/abs/2311.06184>`__
   * - FITS
     - Frequency interpolation, very low parameter count
     - `Xu et al., ICLR 2024 <https://openreview.net/forum?id=bWcnvZ3qMb>`__
   * - SparseTSF
     - Period-aligned downsampling before linear projection
     - `Han et al., ICML 2024 <https://arxiv.org/abs/2405.00946>`__
   * - TimeMixer
     - Past decomposable mixing at multiple time scales
     - `Wang et al., ICLR 2024 <https://openreview.net/forum?id=7oLshfEIC2>`__
   * - NBEATS
     - Neural basis expansion, doubly-residual stacking
     - `Oreshkin et al., ICLR 2020 <https://openreview.net/forum?id=r1ecqn4YwB>`__
   * - NHiTS
     - Hierarchical interpolation, geometrically-increasing pools
     - `Challu et al., AAAI 2023 <https://ojs.aaai.org/index.php/AAAI/article/view/26253>`__
   * - PatchMixer
     - Patch-based MLP-Mixer
     - —
   * - HDMixer
     - Hierarchical dependency mixer
     - —
   * - CycleNet
     - Learnable periodic cycle buffer + residual backbone
     - —
   * - FilterNet
     - Learnable frequency filter bank
     - —
   * - GatedMLPForecaster
     - Gated MLP with channel mixing
     - —
   * - FourierMixerForecaster
     - Fourier-domain MLP mixing
     - —
   * - HarmonicForecaster
     - Harmonic regression basis
     - —
   * - AdaptiveSpectralForecaster
     - Adaptive spectral basis selection
     - —
   * - RandomFourierForecaster
     - Random Fourier feature approximation
     - —
   * - DualDecompForecaster
     - Dual trend-seasonal decomposition
     - —
   * - PrototypicalForecaster
     - Prototype-based feature aggregation
     - —
   * - ImplicitNeuralForecaster
     - Implicit neural representations (INR)
     - —
   * - NeuralBasisForecaster
     - Learnable basis function expansion
     - —
   * - KANForecaster
     - Kolmogorov-Arnold networks
     - —
   * - DishTS
     - Distribution shift-aware normalisation
     - —

**CNN / TCN family**

.. list-table::
   :widths: 22 50 28
   :header-rows: 1

   * - Model
     - Key idea
     - Reference
   * - SCINet
     - Hierarchical downsample-interact-upsample conv tree
     - `Liu et al., NeurIPS 2022 <https://proceedings.neurips.cc/paper_files/paper/2022/hash/266983d0949aed78a16fa4782237dea7-Abstract-Conference.html>`__
   * - TimesNet
     - 2-D temporal variation via TimesBlock + 2-D conv
     - `Wu et al., ICLR 2023 <https://openreview.net/pdf?id=ju_Uqw384Oq>`__
   * - MICN
     - Multi-scale isometric convolution network
     - `Wang et al., ICLR 2023 <https://openreview.net/forum?id=zt53IDUR1U>`__
   * - ModernTCN
     - Modern temporal convolutional network
     - —
   * - WaveNet
     - Dilated causal convolutions with gated activations
     - `van den Oord et al., 2016 <https://arxiv.org/abs/1609.03499>`__
   * - TCNForecaster
     - Vanilla TCN with residual connections
     - —
   * - MultiscaleConvForecaster
     - Parallel multi-scale conv branches
     - —
   * - SincNetForecaster
     - SincNet learnable band-pass filters
     - —
   * - WaveletForecaster
     - Learnable wavelet filter bank
     - —
   * - TemporalConvAttentionForecaster
     - TCN + attention hybrid
     - —

**RNN / SSM / Hybrid family**

.. list-table::
   :widths: 22 50 28
   :header-rows: 1

   * - Model
     - Key idea
     - Reference
   * - RNNForecaster
     - Vanilla LSTM / GRU
     - —
   * - BiLSTMForecaster
     - Bidirectional LSTM
     - —
   * - SegRNN
     - Segment-based RNN (IMO decoding strategy)
     - `Lin et al., ICLR 2024 <https://openreview.net/forum?id=jeqE7rqz2L>`__
   * - Koopa
     - Koopman operator + Fourier learnable dynamics
     - `Liu et al., NeurIPS 2023 <https://arxiv.org/abs/2305.18803>`__
   * - SOFTS
     - Scalable O(1) STAr attention
     - `Han et al., NeurIPS 2024 <https://arxiv.org/abs/2404.04997>`__
   * - MambaForecaster
     - Selective state space (Mamba SSM)
     - `Gu & Dao, 2023 <https://arxiv.org/abs/2312.00752>`__
   * - iMamba
     - Inverted Mamba (channels as tokens)
     - —
   * - SMamba
     - Spatial Mamba for multivariate series
     - —
   * - S4Forecaster
     - Structured state space (S4)
     - `Gu et al., ICLR 2022 <https://openreview.net/forum?id=uYLFoz1vlAC>`__
   * - LRUForecaster
     - Linear recurrent unit
     - —
   * - MinGRUForecaster
     - Minimal gated recurrent unit
     - —
   * - xLSTMForecaster
     - Extended LSTM with exponential gating
     - —
   * - QRNNForecaster
     - Quasi-recurrent neural network
     - —
   * - HGRN2Forecaster
     - Hierarchical gated recurrent network v2
     - —
   * - RWKVForecaster
     - RWKV linear attention / recurrent hybrid
     - —
   * - TSReservoir
     - Echo state / reservoir computing
     - —
   * - EchoStateForecaster
     - Vanilla echo state network
     - —
   * - LiquidNetForecaster
     - Liquid neural networks (adaptive ODE)
     - —
   * - HyenaForecaster
     - Hyena long-convolution operator
     - —
   * - SpikeForecaster
     - Spiking neural network forecaster
     - —

**Graph / Attention variants**

.. list-table::
   :widths: 22 50 28
   :header-rows: 1

   * - Model
     - Key idea
     - Reference
   * - GATForecaster
     - Graph attention network on channel graph
     - —
   * - GCNForecaster
     - Graph convolutional network on channel graph
     - —
   * - GLAForecaster
     - Gated linear attention
     - —
   * - RetForecaster
     - Retentive network (retention mechanism)
     - —
   * - TFT
     - Temporal Fusion Transformer with variable selection
     - —
   * - MoEForecaster
     - Mixture-of-experts gating
     - —
   * - CARD
     - Channel-aligned robust dual-branch Transformer
     - —

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: iTransformer
      :link: ../generated/torch_timeseries.model.iTransformer
      :link-type: doc

      Inverts the token dimension — each *variable* becomes one token.
      Attention captures cross-variate correlations; FFN encodes temporal.

      :bdg-primary:`Forecast` :bdg-secondary:`Impute` :bdg-warning:`Anomaly` :bdg-success:`Classify`

      +++
      `Liu et al., ICLR 2024 <https://openreview.net/forum?id=JePfAI8fah>`__

   .. grid-item-card:: PatchTST
      :link: ../generated/torch_timeseries.model.PatchTST
      :link-type: doc

      Splits the time series into patches, embeds each as a token, then
      applies a standard Transformer encoder. Channel-independent by default.

      :bdg-primary:`Forecast` :bdg-secondary:`Impute` :bdg-warning:`Anomaly` :bdg-success:`Classify`

      +++
      `Nie et al., ICLR 2023 <https://openreview.net/forum?id=Jbdc0vTOcol>`__

   .. grid-item-card:: TimeMixer
      :link: ../generated/torch_timeseries.model.TimeMixer
      :link-type: doc

      Past Decomposable Mixing at multiple scales with Future Multipredictor
      Mixing (FMM) for aggregation.

      :bdg-primary:`Forecast` :bdg-secondary:`Impute` :bdg-warning:`Anomaly` :bdg-success:`Classify`

      +++
      `Wang et al., ICLR 2024 <https://openreview.net/forum?id=7oLshfEIC2>`__

   .. grid-item-card:: DLinear
      :link: ../generated/torch_timeseries.model.DLinear
      :link-type: doc

      Decomposes input into trend + seasonal components, then applies an
      independent linear projection to each.

      :bdg-primary:`Forecast` :bdg-secondary:`Impute` :bdg-warning:`Anomaly` :bdg-success:`Classify`

      +++
      `Zeng et al., AAAI 2023 <https://ojs.aaai.org/index.php/AAAI/article/view/26317>`__

   .. grid-item-card:: Autoformer
      :link: ../generated/torch_timeseries.model.Autoformer
      :link-type: doc

      Auto-Correlation mechanism discovers period-based dependencies via FFT.
      Progressive seasonal-trend decomposition in every encoder/decoder layer.

      :bdg-primary:`Forecast` :bdg-secondary:`Impute` :bdg-warning:`Anomaly` :bdg-success:`Classify`

      +++
      `Wu et al., NeurIPS 2021 <https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html>`__

   .. grid-item-card:: TimesNet
      :link: ../generated/torch_timeseries.model.TimesNet
      :link-type: doc

      Transforms 1-D time series into 2-D feature maps by exploiting
      multi-period structure, then applies 2-D convolutions (TimesBlock).

      :bdg-primary:`Forecast` :bdg-secondary:`Impute` :bdg-warning:`Anomaly` :bdg-success:`Classify`

      +++
      `Wu et al., ICLR 2023 <https://openreview.net/pdf?id=ju_Uqw384Oq>`__

   .. grid-item-card:: MambaForecaster
      :link: ../generated/torch_timeseries.model.MambaForecaster
      :link-type: doc

      Selective State Space Model (Mamba). Linear-time recurrence with
      selective input gates for long-range dependency modelling.

      :bdg-primary:`Forecast` :bdg-secondary:`Impute` :bdg-warning:`Anomaly` :bdg-success:`Classify`

      +++
      `Gu & Dao, 2023 <https://arxiv.org/abs/2312.00752>`__

   .. grid-item-card:: SegRNN
      :link: ../generated/torch_timeseries.model.SegRNN
      :link-type: doc

      Segment-based RNN with IMO decoding. Achieves strong long-horizon
      accuracy without attention.

      :bdg-primary:`Forecast` :bdg-secondary:`Impute` :bdg-warning:`Anomaly` :bdg-success:`Classify`

      +++
      `Lin et al., ICLR 2024 <https://openreview.net/forum?id=jeqE7rqz2L>`__


----

Generation Models
-----------------

Generation models learn to synthesise new time-series windows from scratch.
All expose a ``generate(n, device)`` method returning ``(n, seq_len, n_features)``
tensors and a ``loss(x)`` method for training.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: TimeGAN
      :link: ../generated/torch_timeseries.model.TimeGAN
      :link-type: doc

      GAN with a supervised loss that aligns the stepwise dynamics of the
      generator with real data. Embeds sequences before adversarial training.

      :bdg-info:`Generation`

      +++
      `Yoon et al., NeurIPS 2019 <https://proceedings.neurips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html>`__

   .. grid-item-card:: CSDI
      :link: ../generated/torch_timeseries.model.CSDI
      :link-type: doc

      Conditional Score-based Diffusion. A Transformer denoiser conditioned
      on observed values; handles both generation and imputation.

      :bdg-info:`Generation`

      +++
      `Tashiro et al., NeurIPS 2021 <https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html>`__

   .. grid-item-card:: DiffusionTS
      :link: ../generated/torch_timeseries.model.DiffusionTS
      :link-type: doc

      DDPM denoiser built on trend-seasonal decomposition. Separate Fourier
      and trend branches are fused before the final prediction.

      :bdg-info:`Generation`

      +++
      `Yuan & Qiao, ICLR 2024 <https://openreview.net/forum?id=4h1apFjO99>`__

   .. grid-item-card:: TimeDiff
      :link: ../generated/torch_timeseries.model.TimeDiff
      :link-type: doc

      Self-guided DDPM — a future-mix conditioning strategy guides denoising
      without requiring a separate conditioning signal at inference.

      :bdg-info:`Generation`

      +++
      `Shen & Kwok, ICML 2023 <https://proceedings.mlr.press/v202/shen23d.html>`__

   .. grid-item-card:: NsDiff
      :link: ../generated/torch_timeseries.model.NsDiff
      :link-type: doc

      Non-Stationary Diffusion. Adapts the noise schedule to local variance
      via ``gx`` (rolling standard deviation), denoising with a bidirectional
      GRU conditioned on ``[y_t ‖ y₀_hat ‖ gx]``.

      :bdg-info:`Generation`

      +++
      Ye et al. (preprint)

   .. grid-item-card:: TMDM
      :link: ../generated/torch_timeseries.model.TMDM
      :link-type: doc

      Temporal Diffusion Model. Uses a GRU prior-mean network (``_MuNet``)
      to initialise the reverse process, conditioning denoising on
      ``[y_t ‖ y₀_hat]``.

      :bdg-info:`Generation`

      +++
      Ye et al. (preprint)


----

Irregular Time Series Models
-----------------------------

Irregular models handle asynchronously sampled sequences — each sample may have a
different number of observations and arbitrary timestamps.  All irregular models
accept ``(x, t, mask)`` inputs and optionally a ``t_query`` tensor for seq2seq
interpolation / forecasting output.

Install optional extras once for LatentODE / NeuralCDE / Raindrop:

.. code-block:: bash

   pip install "torch-timeseries[irregular]"

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: GRU-D
      :link: ../generated/torch_timeseries.model.GRUD
      :link-type: doc

      Gated Recurrent Unit with exponential temporal decay. Missing values
      are replaced by a decay-weighted interpolation between the last
      observed value and the feature mean. Supports seq2seq mode via
      ``t_query``.

      :bdg-success:`Irregular Classify` :bdg-primary:`Interpolation` :bdg-warning:`Irregular Forecast`

      +++
      `Che et al., Scientific Reports 2018 <https://www.nature.com/articles/s41598-018-24271-9>`__

   .. grid-item-card:: mTAN
      :link: ../generated/torch_timeseries.model.mTAN
      :link-type: doc

      Multi-Time Attention Network. Learns a small set of reference time
      points and uses cross-attention to encode arbitrary observation
      schedules into a fixed-size representation. No external dependencies.

      :bdg-success:`Irregular Classify` :bdg-primary:`Interpolation` :bdg-warning:`Irregular Forecast`

      +++
      `Shukla & Marlin, ICLR 2021 <https://openreview.net/forum?id=4c0J6lwQ4_>`__

   .. grid-item-card:: LatentODE
      :link: ../generated/torch_timeseries.model.LatentODE
      :link-type: doc

      Variational Latent ODE. A GRU encoder infers the initial latent
      state; an ODE solver evolves it continuously; a decoder reads off
      predictions at any query times. Requires ``torchdiffeq``.

      :bdg-success:`Irregular Classify` :bdg-primary:`Interpolation` :bdg-warning:`Irregular Forecast`

      +++
      `Rubanova et al., NeurIPS 2019 <https://proceedings.neurips.cc/paper/2019/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html>`__

   .. grid-item-card:: NeuralCDE
      :link: ../generated/torch_timeseries.model.NeuralCDE
      :link-type: doc

      Neural Controlled Differential Equation. Fits a natural cubic
      spline to the observations, then solves a CDE driven by the spline
      path to produce a terminal hidden state. Requires ``torchcde``.

      :bdg-success:`Irregular Classify`

      +++
      `Kidger et al., NeurIPS 2020 <https://proceedings.neurips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html>`__

   .. grid-item-card:: Raindrop
      :link: ../generated/torch_timeseries.model.Raindrop
      :link-type: doc

      Graph-guided irregular sensor networks. Per-feature GRUs capture
      local dynamics; a Graph Attention Network models inter-sensor
      dependencies learned from observation patterns. Requires
      ``torch_geometric``.

      :bdg-success:`Irregular Classify`

      +++
      `Zhang et al., ICLR 2022 <https://openreview.net/forum?id=Kwm8I7dU-l5>`__


.. rubric:: Individual model pages (autogenerated)

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_timeseries.model.forecasting_models %}
     {{ name }}
   {% endfor %}
   {% for name in torch_timeseries.model.generation_models %}
     {{ name }}
   {% endfor %}
   {% for name in torch_timeseries.model.irregular_models %}
     {{ name }}
   {% endfor %}
