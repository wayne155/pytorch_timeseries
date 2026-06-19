"""MICN: Multi-scale Isometric Convolution Network for time-series forecasting.

Reference: Wang et al., "MICN: Multi-scale Isometric Convolution Network for
Long-term Time Series Forecasting", ICLR 2023.

Key idea:
  Classic 1-D convolutions on time series struggle to capture both local and
  global patterns simultaneously.  MICN addresses this with two complementary
  branches that together form an *isometric* (same-length-in, same-length-out)
  multi-scale decomposition:

    1. **Seasonal Prediction Block (SPB)**: a *stack* of convolution blocks
       operating at multiple receptive-field scales.  Each scale uses a
       downsampling conv → trend subtraction → upsampling conv pipeline, so
       every branch sees a different temporal resolution.  The branch outputs
       are averaged into a multi-scale seasonal representation.

    2. **Trend Prediction Block (TPB)**: a simple regression on the low-
       frequency trend extracted by averaging over time.

  Final forecast = seasonal component + trend component.

Simplified implementation:
  - SPB: `num_scales` conv blocks with increasing dilation factors (1, 2, …),
    so receptive fields grow without parameter explosion.
  - Each conv block uses a depthwise-separable "isometric" convolution
    (kernel_size, same padding) to preserve sequence length.
  - Trend block: linear regression from (B, T, C) → (B, pred_len, C) applied
    to the input's temporal mean trend.
  - RevIN normalisation on input/output.

Args:
    seq_len:    input lookback window.
    pred_len:   forecast horizon.
    enc_in:     number of variates.
    d_model:    channel dimension of conv blocks.
    num_scales: number of multi-scale branches.
    kernel_size: base convolution kernel size.
    dropout:    dropout rate.
    revin:      use RevIN instance normalisation.
"""
from __future__ import annotations

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
            self._mean = x.mean(1, keepdim=True)
            self._std = x.std(1, keepdim=True).clamp(self.eps)
            x = (x - self._mean) / self._std
            return x * self.affine_weight + self.affine_bias
        else:  # denorm
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            return x * self._std + self._mean


class _IsometricConvBlock(nn.Module):
    """Single-scale seasonal block: dilated depthwise conv + pointwise proj."""

    def __init__(self, enc_in: int, d_model: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.input_proj = nn.Linear(enc_in, d_model)
        self.dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=pad, dilation=dilation, groups=d_model,
        )
        self.pw_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, enc_in)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        h = self.input_proj(x)                          # (B, T, d_model)
        h = h.transpose(1, 2)                           # (B, d_model, T)
        h = F.gelu(self.pw_conv(F.gelu(self.dw_conv(h))))
        h = h.transpose(1, 2)                           # (B, T, d_model)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.norm(h)
        return self.output_proj(h)                      # (B, T, C)


class _SeasonalPredictionBlock(nn.Module):
    """Multi-scale seasonal block: average of isometric conv branches."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, d_model: int,
                 num_scales: int, kernel_size: int, dropout: float):
        super().__init__()
        self.pred_len = pred_len
        # One conv block per scale (increasing dilation)
        self.branches = nn.ModuleList([
            _IsometricConvBlock(enc_in, d_model, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(num_scales)
        ])
        # Project from seq_len → pred_len per-channel
        self.time_proj = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        branch_outs = [b(x) for b in self.branches]           # each (B, T, C)
        seasonal = torch.stack(branch_outs, dim=0).mean(0)    # (B, T, C)
        # project time axis T → pred_len
        out = self.time_proj(seasonal.transpose(1, 2))         # (B, C, pred_len)
        return out.transpose(1, 2)                             # (B, pred_len, C)


class _TrendPredictionBlock(nn.Module):
    """Simple linear trend: pool temporal mean → linear → pred_len."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int):
        super().__init__()
        self.proj = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        # Use channel-wise linear projection on the temporal axis
        out = self.proj(x.transpose(1, 2))     # (B, C, pred_len)
        return out.transpose(1, 2)             # (B, pred_len, C)


class MICN(nn.Module):
    """Multi-scale Isometric Convolution Network (MICN).

    Input  → RevIN → Seasonal + Trend blocks → sum → RevIN-denorm → output
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        num_scales: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.05,
        revin: bool = True,
    ):
        super().__init__()
        self.pred_len = pred_len
        self._revin_flag = revin
        if revin:
            self.revin_layer = _RevIN(enc_in)

        self.spb = _SeasonalPredictionBlock(
            seq_len, pred_len, enc_in, d_model, num_scales, kernel_size, dropout
        )
        self.tpb = _TrendPredictionBlock(seq_len, pred_len, enc_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)
        Returns:
            (B, pred_len, C)
        """
        if self._revin_flag:
            x = self.revin_layer(x, "norm")

        seasonal = self.spb(x)   # (B, pred_len, C)
        trend = self.tpb(x)      # (B, pred_len, C)
        out = seasonal + trend

        if self._revin_flag:
            out = self.revin_layer(out, "denorm")
        return out
