torch_timeseries.model
======================

25 built-in models covering forecasting, generation, and irregular time series.
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

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: DLinear
      :link: ../generated/torch_timeseries.model.DLinear
      :link-type: doc

      Decomposes input into trend + seasonal components, then applies an
      independent linear projection to each. Matches many Transformer
      baselines at a fraction of the cost.

      :bdg-primary:`Forecast` :bdg-secondary:`Impute` :bdg-warning:`Anomaly` :bdg-success:`Classify`

      +++
      `Zeng et al., AAAI 2023 <https://ojs.aaai.org/index.php/AAAI/article/view/26317>`__

   .. grid-item-card:: NLinear
      :link: ../generated/torch_timeseries.model.NLinear
      :link-type: doc

      Subtracts the last time step before projection (normalization trick).
      Extremely lightweight baseline — often hard to beat on univariate data.

      :bdg-primary:`Forecast` :bdg-secondary:`Impute` :bdg-warning:`Anomaly` :bdg-success:`Classify`

      +++
      `Zeng et al., AAAI 2023 <https://ojs.aaai.org/index.php/AAAI/article/view/26317>`__

   .. grid-item-card:: Informer
      :link: ../generated/torch_timeseries.model.Informer
      :link-type: doc

      ProbSparse self-attention reduces complexity to *O(L log L)*.
      Distilling encoder layers progressively compress the sequence.

      :bdg-primary:`Forecast` :bdg-secondary:`Impute` :bdg-warning:`Anomaly` :bdg-success:`Classify`

      +++
      `Zhou et al., AAAI 2021 <https://ojs.aaai.org/index.php/AAAI/article/view/17325>`__

   .. grid-item-card:: Autoformer
      :link: ../generated/torch_timeseries.model.Autoformer
      :link-type: doc

      Auto-Correlation mechanism discovers period-based dependencies via FFT.
      Progressive seasonal-trend decomposition in every encoder/decoder layer.

      :bdg-primary:`Forecast` :bdg-secondary:`Impute` :bdg-warning:`Anomaly` :bdg-success:`Classify`

      +++
      `Wu et al., NeurIPS 2021 <https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html>`__

   .. grid-item-card:: FEDformer
      :link: ../generated/torch_timeseries.model.FEDformer
      :link-type: doc

      Frequency-enhanced decomposed Transformer. Attention and mixing
      are performed in the Fourier / Wavelet domain for *O(L)* complexity.

      :bdg-primary:`Forecast` :bdg-secondary:`Impute` :bdg-warning:`Anomaly` :bdg-success:`Classify`

      +++
      `Zhou et al., ICML 2022 <https://proceedings.mlr.press/v162/zhou22g.html>`__

   .. grid-item-card:: PatchTST
      :link: ../generated/torch_timeseries.model.PatchTST
      :link-type: doc

      Splits the time series into patches, embeds each as a token, then
      applies a standard Transformer encoder. Channel-independent by default.

      :bdg-primary:`Forecast` :bdg-secondary:`Impute` :bdg-warning:`Anomaly` :bdg-success:`Classify`

      +++
      `Nie et al., ICLR 2023 <https://openreview.net/forum?id=Jbdc0vTOcol>`__

   .. grid-item-card:: iTransformer
      :link: ../generated/torch_timeseries.model.iTransformer
      :link-type: doc

      Inverts the token dimension — each *variable* becomes one token.
      Attention captures cross-variate correlations; FFN encodes temporal.

      :bdg-primary:`Forecast` :bdg-secondary:`Impute` :bdg-warning:`Anomaly` :bdg-success:`Classify`

      +++
      `Liu et al., ICLR 2024 <https://openreview.net/forum?id=JePfAI8fah>`__

   .. grid-item-card:: TSMixer
      :link: ../generated/torch_timeseries.model.TSMixer
      :link-type: doc

      MLP-Mixer architecture alternating time-mixing and feature-mixing
      layers. No attention — pure MLP with residual connections.

      :bdg-primary:`Forecast` :bdg-secondary:`Impute` :bdg-warning:`Anomaly` :bdg-success:`Classify`

      +++
      `Chen et al., KDD 2023 <https://arxiv.org/abs/2303.06053>`__

   .. grid-item-card:: Crossformer
      :link: ../generated/torch_timeseries.model.Crossformer
      :link-type: doc

      Two-Stage Attention Router: cross-time attention on patches, then
      cross-dimension attention to model inter-variable dependencies.

      :bdg-primary:`Forecast` :bdg-secondary:`Impute` :bdg-warning:`Anomaly` :bdg-success:`Classify`

      +++
      `Zhang & Yan, ICLR 2023 <https://openreview.net/forum?id=vSVLM2j9eie>`__

   .. grid-item-card:: SCINet
      :link: ../generated/torch_timeseries.model.SCINet
      :link-type: doc

      Hierarchical downsample-interact-upsample tree. Each node applies
      sample convolution on the odd/even sub-sequences.

      :bdg-primary:`Forecast` :bdg-secondary:`Impute` :bdg-warning:`Anomaly` :bdg-success:`Classify`

      +++
      `Liu et al., NeurIPS 2022 <https://proceedings.neurips.cc/paper_files/paper/2022/hash/266983d0949aed78a16fa4782237dea7-Abstract-Conference.html>`__

   .. grid-item-card:: TimesNet
      :link: ../generated/torch_timeseries.model.TimesNet
      :link-type: doc

      Transforms 1-D time series into 2-D feature maps by exploiting
      multi-period structure, then applies 2-D convolutions (TimesBlock).

      :bdg-primary:`Forecast` :bdg-secondary:`Impute` :bdg-warning:`Anomaly` :bdg-success:`Classify`

      +++
      `Wu et al., ICLR 2023 <https://openreview.net/pdf?id=ju_Uqw384Oq>`__

   .. grid-item-card:: CATS
      :link: ../generated/torch_timeseries.model.CATS
      :link-type: doc

      Contiguous Adaptive Time Series: extends the context window by
      generating auxiliary future queries, improving long-horizon accuracy.

      :bdg-primary:`Forecast`

      +++
      `Lin et al., ICML 2024 <https://arxiv.org/abs/2403.01673>`__

   .. grid-item-card:: FITS
      :link: ../generated/torch_timeseries.model.FITS
      :link-type: doc

      Frequency Interpolation: compress the look-back window in the Fourier
      domain, then interpolate to the prediction horizon — very few params.

      :bdg-primary:`Forecast` :bdg-secondary:`Impute` :bdg-warning:`Anomaly` :bdg-success:`Classify`

      +++
      `Xu et al., ICLR 2024 <https://openreview.net/forum?id=bWcnvZ3qMb>`__

   .. grid-item-card:: FreTS
      :link: ../generated/torch_timeseries.model.FreTS
      :link-type: doc

      All-MLP in the frequency domain. Real and imaginary components are
      mixed separately before converting back to the time domain.

      :bdg-primary:`Forecast` :bdg-secondary:`Impute` :bdg-warning:`Anomaly` :bdg-success:`Classify`

      +++
      `Yi et al., NeurIPS 2023 <https://proceedings.neurips.cc/paper_files/paper/2023/hash/f1d16af76939f476b5f040fd1398c0a3-Abstract-Conference.html>`__


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
