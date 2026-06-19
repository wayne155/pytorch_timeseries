"""GatedMLPForecaster: Gated MLP (gMLP) with Spatial Gating Units for forecasting.

Key idea (Liu et al., 2021 — "Pay Attention to MLPs"):

Replace multi-head self-attention with a *Spatial Gating Unit* (SGU) that
gates feature activations using a linear projection across the time axis:

    u, v = split(Linear(LN(x)), dim=-1)    two halves of the projected features
    gate = W_s · LN(v)                    (W_s ∈ ℝ^{T×T}) mixes across time
    z    = u ⊙ gate                        element-wise gate
    y    = Linear(z) + x                   residual output projection

The spatial gate W_s is a dense T×T linear map that, unlike attention,
  - has no softmax normalisation
  - has no query/key dot-product
  - is the same for every example (input-independent)
  - mixes all time steps simultaneously with O(T²) parameters

This fills a different niche from every existing model in the library:
  - TSMixer: independent temporal-mix + channel-mix linear layers.
  - Transformers: data-dependent softmax(QK^T/√d) V attention.
  - Linear (DLinear): single global linear per channel.
  - GatedMLPForecaster: gated split-mix with a *position-mixing linear*.

Architecture::

    Input (B, T, C)
      → RevIN norm
      → CI reshape → (B·C, T)
      → Linear embed → (B·C, T, d_model)
      → n_layers × gMLPBlock:
            LN → Linear(d_model, 2·d_ffn) → GELU → u, v split
            v → LN → W_spatial(T→T) → gate
            z = u ⊙ gate
            Linear(d_ffn, d_model) → residual
      → global mean-pool → (B·C, d_model)
      → Linear head → (B·C, pred_len)
      → reshape → (B, pred_len, C)
      → RevIN denorm

Args:
    seq_len:   input lookback T.
    pred_len:  forecast horizon.
    enc_in:    number of variates C.
    d_model:   model width.
    d_ffn:     gated projection width (each of the two halves).
    n_layers:  number of gMLP blocks.
    dropout:   dropout probability.
    revin:     use RevIN instance normalisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.revin import RevIN


class _SGU(nn.Module):
    """Spatial Gating Unit: linearly mixes across time dimension."""

    def __init__(self, seq_len: int, d_ffn: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        # Dense T × T spatial mixing matrix (initialise near identity for stability)
        self.W_s = nn.Linear(seq_len, seq_len, bias=True)
        nn.init.normal_(self.W_s.weight, std=1e-4)
        with torch.no_grad():
            self.W_s.weight.add_(torch.eye(seq_len))   # near-identity init
        self.drop = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """u, v: (B, T, d_ffn) → (B, T, d_ffn) gated output."""
        # Mix across T dimension
        gate = self.W_s(self.norm(v).permute(0, 2, 1)).permute(0, 2, 1)   # (B, T, d_ffn)
        return self.drop(u * gate)


class _gMLPBlock(nn.Module):
    """One gMLP block: channel expansion → spatial gating → residual."""

    def __init__(self, d_model: int, d_ffn: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj_in = nn.Linear(d_model, 2 * d_ffn)
        self.sgu = _SGU(seq_len, d_ffn, dropout)
        self.proj_out = nn.Linear(d_ffn, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        h = F.gelu(self.proj_in(self.norm(x)))       # (B, T, 2*d_ffn)
        u, v = h.chunk(2, dim=-1)                    # each (B, T, d_ffn)
        z = self.sgu(u, v)                           # (B, T, d_ffn)
        return x + self.drop(self.proj_out(z))


class GatedMLPForecaster(nn.Module):
    """Gated MLP (gMLP) forecaster with spatial gating units (channel-independent)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        d_ffn: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
        revin: bool = True,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in
        self._revin_flag = revin

        if revin:
            self.revin_layer = RevIN(enc_in, affine=True)

        self.embed = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList(
            [_gMLPBlock(d_model, d_ffn, seq_len, dropout) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, pred_len, C)"""
        B, T, C = x.shape

        if self._revin_flag:
            x = self.revin_layer(x, "norm")

        x_ci = x.permute(0, 2, 1).reshape(B * C, T, 1)   # (BC, T, 1)
        x_ci = self.embed(x_ci)                            # (BC, T, d_model)

        for block in self.blocks:
            x_ci = block(x_ci)

        ctx = x_ci.mean(dim=1)                             # (BC, d_model)
        pred = self.head(ctx)                              # (BC, pred_len)
        pred = pred.reshape(B, C, self.pred_len).permute(0, 2, 1)

        if self._revin_flag:
            pred = self.revin_layer(pred, "denorm")
        return pred
