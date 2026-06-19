"""Koopa: Koopman operator-based non-stationary time-series forecasting.

Reference: Liu et al., "Koopa: Learning Non-stationary Time Series Dynamics
with Koopman Predictors", NeurIPS 2023.

Key idea:
  Non-stationary time series are difficult for models that assume stationarity.
  Koopa decomposes the series into:
    1. **Time-variant (trend)** component — low-frequency, non-periodic.
    2. **Time-invariant (periodic)** component — seasonal / cyclic.

  Each component is propagated using a learned **Koopman operator** (a linear
  map K in a lifted feature space), enabling exact multi-step prediction via
  repeated operator application: ĥ_{t+n} = Kⁿ ĥ_t.

Architecture:
  1. Decompose x into trend + seasonal via moving-average filter.
  2. **Fourier Filter**: isolate seasonal component in frequency domain;
     keep only the top-k Fourier frequencies (channel-independent).
  3. **Koopman Forecaster** per component:
       a. Encoder MLP: x_seg → hidden representation h.
       b. Koopman operator K (square linear, d×d) applied iteratively.
       b. Decoder MLP: h → forecast segment.
  4. Reassemble: final forecast = trend_forecast + seasonal_forecast.

Implementation choices:
  * Multi-step forecasting via iterative operator application.
  * Each segment covers `seg_len` time steps; operator is applied once per
    segment to produce the next segment.
  * Both decomposed components share the same operator architecture but use
    separate parameter sets.
  * RevIN normalisation applied before decomposition (recommended).

Args:
    seq_len:    input lookback window.
    pred_len:   forecasting horizon.
    enc_in:     number of variates (channels).
    seg_len:    segment length for the Koopman predictor (default 10).
    d_model:    hidden dimension of the Koopman lifting space.
    n_ff:       MLP hidden size for encoder/decoder (default 4×d_model).
    top_k:      number of Fourier frequencies retained for seasonal component.
    revin:      use Reversible Instance Normalisation.
    dropout:    dropout in encoder/decoder MLPs.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True)
            self._std = x.std(dim=1, keepdim=True) + self.eps
            x = (x - self._mean) / self._std
            x = x * self.affine_weight + self.affine_bias
        elif mode == "denorm":
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self._std + self._mean
        return x


class _FourierFilter(nn.Module):
    """Retain only the top-k Fourier frequencies (channel-independent).

    This isolates the periodic (time-invariant) component; the residual
    is the trend (time-variant) component.
    """

    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``(B, T, N)``

        Returns:
            seasonal: ``(B, T, N)`` — low-rank Fourier reconstruction
            trend:    ``(B, T, N)`` — residual
        """
        B, T, N = x.shape
        # FFT over the time dimension
        xf = torch.fft.rfft(x, dim=1)              # (B, T//2+1, N)

        # Keep only the top-k amplitudes (channel-independent selection)
        amp = xf.abs()                              # (B, T//2+1, N)
        # Select top-k frequencies by mean amplitude across batch and channels
        mean_amp = amp.mean(dim=(0, 2))             # (T//2+1,)
        k = min(self.top_k, mean_amp.size(0))
        _, top_idx = mean_amp.topk(k, largest=True)

        mask = torch.zeros_like(xf)
        mask[:, top_idx, :] = 1.0
        xf_seasonal = xf * mask

        seasonal = torch.fft.irfft(xf_seasonal, n=T, dim=1)  # (B, T, N)
        trend = x - seasonal
        return seasonal, trend


class _KoopmanPredictor(nn.Module):
    """Encoder → Koopman operator (iterated) → Decoder.

    Input and output are both in segment space: each call processes one
    segment of length `seg_len` and predicts one future segment.
    """

    def __init__(self, seg_len: int, enc_in: int, d_model: int, n_ff: int,
                 dropout: float = 0.0):
        super().__init__()
        self.seg_len = seg_len
        flat_in = seg_len * enc_in

        self.encoder = nn.Sequential(
            nn.Linear(flat_in, n_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_ff, d_model),
        )
        # Koopman operator — square linear in lifting space
        self.koopman = nn.Linear(d_model, d_model, bias=False)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, n_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_ff, flat_in),
        )

    def forward(self, x_seg: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        Args:
            x_seg:     ``(B, seg_len, N)`` — the last observed segment.
            num_steps: how many segments to predict.

        Returns:
            ``(B, num_steps * seg_len, N)``
        """
        B, L, N = x_seg.shape
        h = self.encoder(x_seg.reshape(B, -1))     # (B, d_model)

        segments = []
        for _ in range(num_steps):
            h = self.koopman(h)                    # one Koopman step
            seg = self.decoder(h).reshape(B, L, N) # (B, seg_len, N)
            segments.append(seg)

        return torch.cat(segments, dim=1)           # (B, num_steps*seg_len, N)


class Koopa(nn.Module):
    """Koopman operator-based non-stationary time-series forecasting.

    Decomposes the series into seasonal + trend, forecasts each component
    using an independent Koopman predictor, then reassembles.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        seg_len: int = 10,
        d_model: int = 128,
        n_ff: Optional[int] = None,
        top_k: int = 5,
        revin: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.seg_len = seg_len
        n_ff = n_ff if n_ff is not None else d_model * 4

        self.revin = revin
        if revin:
            self.revin_layer = _RevIN(enc_in)

        self.fourier_filter = _FourierFilter(top_k=top_k)

        # Number of Koopman steps = ceil(pred_len / seg_len)
        self.num_steps = math.ceil(pred_len / seg_len)

        self.seasonal_pred = _KoopmanPredictor(seg_len, enc_in, d_model, n_ff, dropout)
        self.trend_pred = _KoopmanPredictor(seg_len, enc_in, d_model, n_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, seq_len, enc_in)``.

        Returns:
            Forecast ``(B, pred_len, enc_in)``.
        """
        if self.revin:
            x = self.revin_layer(x, "norm")

        # Decompose
        seasonal, trend = self.fourier_filter(x)   # each (B, T, N)

        # Use the last segment as the seed for Koopman iteration
        last_seg = x[:, -self.seg_len:, :]         # (B, seg_len, N)

        # Generate num_steps segments per component
        seasonal_fc = self.seasonal_pred(
            seasonal[:, -self.seg_len:, :], self.num_steps
        )                                           # (B, num_steps*seg_len, N)
        trend_fc = self.trend_pred(
            trend[:, -self.seg_len:, :], self.num_steps
        )                                           # (B, num_steps*seg_len, N)

        # Reassemble and trim to pred_len
        out = (seasonal_fc + trend_fc)[:, :self.pred_len, :]   # (B, pred_len, N)

        if self.revin:
            out = self.revin_layer(out, "denorm")

        return out
