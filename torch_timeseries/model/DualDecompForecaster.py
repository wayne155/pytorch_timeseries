"""DualDecompForecaster: Explicit trend/seasonal decomposition with dual branches.

Key idea:
  Time-series can be decomposed as:   x[t] = trend[t] + seasonal[t]

  Standard approaches (Autoformer, ETSformer) embed decomposition *inside* each
  layer.  DualDecompForecaster instead does a *single upfront* decomposition and
  then applies two *specialised* forecasters:

    • Trend branch  — a direct linear projection (trends are low-frequency and
      globally smooth, so a linear readout on the pooled trend component suffices).
    • Seasonal branch — a lightweight Transformer operating on the de-trended signal
      (capturing periodic / non-linear structure).

  The final prediction is the sum of both branches:
      ŷ = trend_head(trend) + seasonal_attn(seasonal)

  Decomposition uses a causal moving-average filter of width `kernel_size`:
      trend[t] = (1/k) Σ_{j=0}^{k-1} x[t-j]      (mean-pooling with padding)
      seasonal[t] = x[t] - trend[t]

  Unlike Autoformer:
    • Decomposition happens once before any encoding, not per-layer.
    • The trend branch is a simple Linear (no attention).
    • The seasonal branch uses standard (softmax) self-attention, not autocorrelation.

Architecture:
  1. RevIN normalise.
  2. Moving-average decomposition: trend, seasonal = decompose(x).
  3. Trend branch:  linear projection  (B*C, T) → (B*C, pred_len).
  4. Seasonal branch:  patch embed → e_layers Transformer layers → mean-pool → head.
  5. Out = trend_out + seasonal_out.

Args:
    seq_len:      lookback T.
    pred_len:     forecast horizon.
    enc_in:       number of variates.
    kernel_size:  moving-average window width (must be odd for symmetric results;
                  clamp to seq_len if larger).
    d_model:      token dimension for seasonal Transformer.
    n_heads:      attention heads.
    e_layers:     seasonal Transformer layers.
    d_ff:         feed-forward hidden size.
    patch_len:    patch size for seasonal tokenisation (default 8).
    dropout:      dropout rate.
    revin:        use RevIN normalisation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# RevIN
# ──────────────────────────────────────────────────────────────────────────────


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
        else:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            return x * self._std + self._mean


# ──────────────────────────────────────────────────────────────────────────────
# Moving-average decomposition
# ──────────────────────────────────────────────────────────────────────────────


class _MovingAvgDecomp(nn.Module):
    """Moving-average (mean-pool) trend + residual seasonal decomposition."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        # Causal moving average: pad left so output has same length as input
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        self.pad = kernel_size - 1   # left padding for causal MA

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B*C, T)
        Returns:
            trend: (B*C, T)
            seasonal: (B*C, T)
        """
        # Pad left by (kernel_size-1) to get causal average with same output length
        x_padded = F.pad(x.unsqueeze(1), (self.pad, 0), mode="replicate")
        trend = self.avg(x_padded).squeeze(1)  # (B*C, T)
        seasonal = x - trend
        return trend, seasonal


# ──────────────────────────────────────────────────────────────────────────────
# Seasonal Transformer (standard softmax attention)
# ──────────────────────────────────────────────────────────────────────────────


class _SeasonalLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        x = x + self.drop(self.ff2(F.gelu(self.ff1(self.norm2(x)))))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# DualDecompForecaster
# ──────────────────────────────────────────────────────────────────────────────


class DualDecompForecaster(nn.Module):
    """Trend/seasonal decomposition forecaster with dual specialised branches."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        kernel_size: int = 25,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 128,
        patch_len: int = 8,
        dropout: float = 0.1,
        revin: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self._revin_flag = revin

        if revin:
            self.revin_layer = _RevIN(enc_in)

        # Clamp kernel_size to valid range
        kernel_size = min(kernel_size, seq_len)
        if kernel_size % 2 == 0:
            kernel_size += 1  # ensure odd for conceptual symmetry
        self.decomp = _MovingAvgDecomp(kernel_size)

        # ── Trend branch: simple linear ───────────────────────────────────────
        self.trend_head = nn.Linear(seq_len, pred_len)

        # ── Seasonal branch: patch Transformer ───────────────────────────────
        n_patches = math.ceil((seq_len - patch_len) / patch_len) + 1
        pad_len = max(0, (n_patches - 1) * patch_len + patch_len - seq_len)
        self.patch_len = patch_len
        self.n_patches = n_patches
        self.pad_len = pad_len

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.seasonal_layers = nn.ModuleList(
            [_SeasonalLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )
        self.seasonal_norm = nn.LayerNorm(d_model)
        self.seasonal_head = nn.Linear(d_model, pred_len)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)
        Returns:
            (B, pred_len, C)
        """
        B, T, C = x.shape

        if self._revin_flag:
            x = self.revin_layer(x, "norm")

        x_ci = x.transpose(1, 2).reshape(B * C, T)              # (BC, T)

        # ── Decompose ─────────────────────────────────────────────────────────
        trend, seasonal = self.decomp(x_ci)                      # each (BC, T)

        # ── Trend branch ──────────────────────────────────────────────────────
        trend_out = self.trend_head(trend)                       # (BC, pred_len)

        # ── Seasonal branch ───────────────────────────────────────────────────
        s = seasonal
        if self.pad_len > 0:
            s = F.pad(s, (0, self.pad_len))
        patches = s.unfold(-1, self.patch_len, self.patch_len)   # (BC, n_patches, patch_len)
        h = self.patch_embed(patches) + self.pos_embed
        h = self.drop(h)
        for layer in self.seasonal_layers:
            h = layer(h)
        h = self.seasonal_norm(h).mean(dim=1)                    # (BC, d_model)
        seasonal_out = self.seasonal_head(h)                     # (BC, pred_len)

        out = trend_out + seasonal_out                           # (BC, pred_len)
        out = out.reshape(B, C, self.pred_len).transpose(1, 2)  # (B, pred_len, C)

        if self._revin_flag:
            out = self.revin_layer(out, "denorm")
        return out
