"""TemporalConvAttentionForecaster: Dilated causal TCN backbone + temporal self-attention.

Key idea:
  Two complementary components:
  1. **Dilated causal TCN** (WaveNet-style) builds local receptive fields:
       each residual block doubles the dilation, giving log-depth global coverage.
  2. **Temporal self-attention** aggregates the resulting per-timestep features
       across the full sequence, allowing the model to weight which positions
       matter most for the forecast.

  The combination — local inductive bias from convolutions, global selection
  from attention — differs from pure-attention (Transformer) and pure-TCN models.

Architecture comparison:
  TCNForecaster:                 dilated causal TCN, last-timestep head, no attention.
  WaveNet:                       gated TCN, skip-sum → head, no attention.
  VanillaTransformer:            full O(T²) attention, no convolution.
  TemporalConvAttentionForecaster: TCN → full T × T self-attention → mean-pool → head.

Architecture::

    Input (B, T, C)
      → RevIN norm
      → CI reshape → (B·C, T)
      → patch embed Linear(T → d_model) channel → (B·C, 1, d_model)  [optional]
      -- OR --
      → 1-D projection Linear(1, d_model) per timestep
      → N × DilatedCausalResBlock: Conv1d(d, d, k, dilation=2^l, causal-pad)
      → multi-head self-attention over T timesteps
      → mean pool + head → (B·C, pred_len)
      → reshape → (B, pred_len, C)
      → RevIN denorm

Args:
    seq_len:    input lookback T.
    pred_len:   forecast horizon.
    enc_in:     number of variates C.
    d_model:    TCN channel width and attention dimension.
    n_heads:    attention heads.
    n_blocks:   number of dilated residual TCN blocks.
    kernel_size: convolution kernel width.
    dropout:    dropout rate.
    revin:      use RevIN instance normalisation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.revin import RevIN


class _CausalDilatedResBlock(nn.Module):
    """Dilated causal 1-D residual convolution block."""

    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation  # left-only causal padding
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=0)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=0)
        self.pad = pad
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_model, T)"""
        # Causal convolution: pad left, no right
        h = F.pad(x, (self.pad, 0))
        h = F.gelu(self.conv1(h))                    # (B, d, T)
        h = h.transpose(1, 2)                         # (B, T, d)
        h = self.norm1(h).transpose(1, 2)             # (B, d, T)

        h = F.pad(h, (self.pad, 0))
        h = self.drop(F.gelu(self.conv2(h)))
        h = h.transpose(1, 2)
        h = self.norm2(h).transpose(1, 2)

        return x + h                                  # residual


class TemporalConvAttentionForecaster(nn.Module):
    """Dilated causal TCN + temporal self-attention (channel-independent)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        revin: bool = True,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in
        self._revin_flag = revin

        if revin:
            self.revin_layer = RevIN(enc_in, affine=True)

        # Project each (scalar) timestep to d_model channels
        self.input_proj = nn.Linear(1, d_model)

        # Dilated causal TCN: dilation doubles each block
        self.tcn_blocks = nn.ModuleList([
            _CausalDilatedResBlock(d_model, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(n_blocks)
        ])

        # Temporal multi-head self-attention over T timesteps
        assert d_model % n_heads == 0
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_attn = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, pred_len)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, pred_len, C)"""
        B, T, C = x.shape

        if self._revin_flag:
            x = self.revin_layer(x, "norm")

        # CI: (BC, T)
        x_ci = x.permute(0, 2, 1).reshape(B * C, T)

        # Project to (BC, T, d_model) then transpose to (BC, d_model, T)
        h = self.input_proj(x_ci.unsqueeze(-1))          # (BC, T, d_model)
        h = h.transpose(1, 2)                            # (BC, d_model, T)

        # Dilated causal TCN
        for block in self.tcn_blocks:
            h = block(h)

        # Back to (BC, T, d_model) for attention
        h = h.transpose(1, 2)                            # (BC, T, d_model)

        # Temporal self-attention (residual + norm)
        attn_out, _ = self.attn(h, h, h)
        h = self.norm_attn(h + self.drop(attn_out))     # (BC, T, d_model)

        # Mean-pool over timesteps → head
        feat = h.mean(dim=1)                             # (BC, d_model)
        pred = self.head(feat)                           # (BC, pred_len)
        pred = pred.reshape(B, C, self.pred_len).permute(0, 2, 1)   # (B, pred_len, C)

        if self._revin_flag:
            pred = self.revin_layer(pred, "denorm")
        return pred
