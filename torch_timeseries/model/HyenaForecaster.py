"""HyenaForecaster: Long-convolution operator with position-conditioned filters.

Key idea (Poli et al., 2023 — "Hyena Hierarchy: Towards Larger Convolutional
Language Models"):

The Hyena operator replaces attention with a *long convolution* whose filter
is not fixed (like S4D) nor data-derived (like Transformers), but is
synthesised by a small MLP conditioned on *positional encodings*:

    h_t = FilterMLP( pos_enc(t) )    ← filter, one value per time step
    y = IFFT( FFT(v) · FFT(h) )      ← sub-quadratic long convolution
    out = y ⊙ gate                   ← multiplicative gate

where `gate = σ(W_g x)` provides the data-dependent modulation, and
`v = W_v x` is the value projection.

This places Hyena in its own architectural niche:

  ┌─────────────────┬──────────────────┬──────────────────────┐
  │  Filter type    │   Model          │   Filter source      │
  ├─────────────────┼──────────────────┼──────────────────────┤
  │  Fixed analytic │  S4Forecaster    │  Vandermonde / SSM   │
  │  Learned linear │  FourierMixer    │  Linear map on freq  │
  │  Data-dependent │  Transformers    │  Q·K attention       │
  │  Position-MLP   │  HyenaForecaster │  MLP(pos_enc(t))     │
  └─────────────────┴──────────────────┴──────────────────────┘

Architecture::

    Input (B, T, C)
      → RevIN norm
      → CI reshape → (B·C, T)
      → Linear embed → (B·C, T, d_model)
      → n_layers × HyenaBlock:
          v       = W_v(x)                    projection
          gate    = σ(W_g(x))                 data-dependent gate
          h[t]    = FilterMLP(pos_enc(t))     position filter
          y       = IFFT(FFT(v)·FFT(h))       long conv (FFT, O(T log T))
          out     = y ⊙ gate                  gated output
          out     = W_o(out) + x              residual + out-projection
          out     = LayerNorm(out)
      → global mean-pool → (B·C, d_model)
      → Linear head → (B·C, pred_len)
      → reshape → (B, pred_len, C)
      → RevIN denorm

Args:
    seq_len:    input lookback T.
    pred_len:   forecast horizon.
    enc_in:     number of variates C.
    d_model:    model width (hidden dimension).
    n_layers:   number of Hyena blocks.
    pos_freqs:  number of sinusoidal frequency bands in the positional encoding.
    filter_dim: hidden width of the filter MLP.
    dropout:    dropout probability.
    revin:      use RevIN instance normalisation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.revin import RevIN


class _PosFilterMLP(nn.Module):
    """Position-conditioned filter generator.

    Maps sinusoidal positional encodings → per-time-step filter values.

        pos_enc(t) = [sin(2πk t/T), cos(2πk t/T)]_{k=1..K}  ∈ ℝ^{2K}
        h[t] = MLP(pos_enc(t))  ∈ ℝ^{d_model}

    The filter is re-generated each forward pass from the (T,) positional
    features so that it adapts to different sequence lengths at inference
    time (though in CI forecasting seq_len is fixed).
    """

    def __init__(self, pos_freqs: int, filter_dim: int, d_model: int, seq_len: int) -> None:
        super().__init__()
        self.d_model = d_model
        pos_in = 2 * pos_freqs + 1   # sin/cos pairs + constant

        self.mlp = nn.Sequential(
            nn.Linear(pos_in, filter_dim),
            nn.SiLU(),
            nn.Linear(filter_dim, filter_dim),
            nn.SiLU(),
            nn.Linear(filter_dim, d_model),
        )
        # Pre-compute positional features (not a learned parameter)
        self.register_buffer("pos_feat", self._make_pos(seq_len, pos_freqs))

    @staticmethod
    def _make_pos(T: int, K: int) -> torch.Tensor:
        """Build (T, 2K+1) positional features."""
        t = torch.arange(T, dtype=torch.float32) / T   # normalised ∈ [0, 1)
        feats = [torch.ones(T)]
        for k in range(1, K + 1):
            feats.append(torch.sin(2 * math.pi * k * t))
            feats.append(torch.cos(2 * math.pi * k * t))
        return torch.stack(feats, dim=-1)   # (T, 2K+1)

    def forward(self) -> torch.Tensor:
        """Returns (T, d_model) filter."""
        return self.mlp(self.pos_feat)   # (T, d_model)


class _HyenaBlock(nn.Module):
    """One Hyena operator block."""

    def __init__(
        self,
        d_model: int,
        pos_freqs: int,
        filter_dim: int,
        seq_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Value + gate projections
        self.proj_v = nn.Linear(d_model, d_model)
        self.proj_g = nn.Linear(d_model, d_model)
        # Output projection
        self.proj_o = nn.Linear(d_model, d_model)

        # Filter MLP (generates the long-conv kernel from positional encodings)
        self.filter_mlp = _PosFilterMLP(pos_freqs, filter_dim, d_model, seq_len)

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        B, T, H = x.shape

        v = self.proj_v(x)                           # (B, T, H)
        gate = torch.sigmoid(self.proj_g(x))         # (B, T, H)

        # Generate position filter: (T, H)
        h_filt = self.filter_mlp()                   # (T, H)

        # Long convolution via FFT: (B, H, T) × (H, T) → (B, H, T)
        n_fft = 2 * T
        V = torch.fft.rfft(v.permute(0, 2, 1), n=n_fft, dim=-1)   # (B, H, f)
        F_h = torch.fft.rfft(h_filt.T, n=n_fft, dim=-1)           # (H, f)
        Y = torch.fft.irfft(V * F_h.unsqueeze(0), n=n_fft, dim=-1)[..., :T]  # (B, H, T)
        y = Y.permute(0, 2, 1)                       # (B, T, H)

        # Gate + output projection + residual
        y = self.drop(y * gate)
        y = self.proj_o(y)
        return self.norm(x + y)


class HyenaForecaster(nn.Module):
    """Hyena long-convolution forecaster (channel-independent)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        n_layers: int = 3,
        pos_freqs: int = 16,
        filter_dim: int = 64,
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
            [_HyenaBlock(d_model, pos_freqs, filter_dim, seq_len, dropout)
             for _ in range(n_layers)]
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
