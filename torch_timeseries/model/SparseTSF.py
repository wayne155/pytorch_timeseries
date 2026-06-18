"""SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters.

Reference: Chen et al., "SparseTSF: Modeling Long-term Time Series Forecasting
with ~1k Parameters", ICML 2024.  https://arxiv.org/abs/2405.00946

Core idea:
  1. Downsample the input series by picking every P-th observation (period P).
  2. Apply a tiny linear layer to each downsampled sub-series independently.
  3. Upsample back to the prediction horizon via reshape + linear interpolation.

This gives ~P * (seq_len/P + pred_len/P) ≈ seq_len + pred_len parameters total,
far fewer than any attention-based model while remaining competitive on ETT benchmarks.

Args:
    seq_len:    input lookback window.
    pred_len:   forecasting horizon.
    enc_in:     number of variates (channel-independent processing).
    period:     downsampling period.  Defaults to ``seq_len // 4`` (quarter-frequency).
    revin:      use reversible instance normalisation (RevIN).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RevIN(nn.Module):
    """Per-channel reversible instance normalisation."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
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


class SparseTSF(nn.Module):
    """SparseTSF: period-based sparse forecasting with minimal parameters.

    Processes each variate independently (channel-independent) by:
      1. Padding the input to the nearest multiple of ``period``.
      2. Downsampling: reshape ``(B, T_padded, N)`` → ``(B * N, period, T_padded//period)``
         then average-pool along the period dimension to get ``(B * N, T_down)`` where
         ``T_down = T_padded // period``.
      3. Apply a linear layer ``T_down → pred_steps`` where
         ``pred_steps = ceil(pred_len / period)``.
      4. Reshape and interpolate back to ``pred_len``.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        period: Optional[int] = None,
        revin: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.period = period if period is not None else max(1, seq_len // 4)
        self.revin = revin

        # Compute downsampled dimensions
        T_pad = math.ceil(seq_len / self.period) * self.period
        self.T_pad = T_pad
        self.T_down = T_pad // self.period
        self.pred_steps = math.ceil(pred_len / self.period)

        # Core learnable component: one weight matrix for all variates
        self.linear = nn.Linear(self.T_down, self.pred_steps)

        if revin:
            self.revin_layer = _RevIN(enc_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, seq_len, enc_in)``.

        Returns:
            Forecast ``(B, pred_len, enc_in)``.
        """
        B, T, N = x.shape

        if self.revin:
            x = self.revin_layer(x, "norm")

        # Pad to multiple of period
        if T < self.T_pad:
            pad_len = self.T_pad - T
            x = F.pad(x, (0, 0, pad_len, 0))   # pad on time-left with zeros

        # Downsample: (B, T_pad, N) → (B * N, T_down)
        # Reshape to (B * N, T_pad), then take every period-th step
        x_perm = x.permute(0, 2, 1)                   # (B, N, T_pad)
        x_flat = x_perm.reshape(B * N, self.T_pad)    # (B*N, T_pad)
        # Average over each period window: reshape to (B*N, T_down, period), mean over period
        x_down = x_flat.reshape(B * N, self.T_down, self.period).mean(dim=-1)  # (B*N, T_down)

        # Linear forecast in sparse domain
        y_down = self.linear(x_down)                   # (B*N, pred_steps)

        # Upsample: repeat each step `period` times, then trim to pred_len
        y_up = y_down.unsqueeze(-1).expand(-1, -1, self.period)  # (B*N, pred_steps, period)
        y_up = y_up.reshape(B * N, self.pred_steps * self.period)  # (B*N, pred_steps*period)
        y_up = y_up[:, : self.pred_len]                # (B*N, pred_len)

        # Reshape back to (B, pred_len, N)
        y = y_up.reshape(B, N, self.pred_len).permute(0, 2, 1)  # (B, pred_len, N)

        if self.revin:
            y = self.revin_layer(y, "denorm")

        return y
