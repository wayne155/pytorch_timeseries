"""PrototypicalForecaster: Memory-augmented prototype network for forecasting.

Key idea:

Learn K prototype temporal patterns in a compact latent space.  At inference
time, encode the input window as a query, retrieve a weighted combination of
prototype-associated predictions, and blend it with a direct MLP forecast:

    query   = QueryEncoder(x)       ∈ ℝ^{d_proto}
    sim_k   = query · key_k / √d   similarity to prototype k
    α_k     = softmax(sim)          soft retrieval weights
    ŷ_mem   = Σ_k α_k val_k        memory-retrieved forecast
    ŷ_dir   = DirectMLP(x)         fallback direct forecast
    ŷ       = OutputMLP([ŷ_mem, ŷ_dir])  fused output

The prototype keys (K × d_proto) and values (K × pred_len) are nn.Parameters
optimised end-to-end.  A temperature parameter τ (learnable, initialised to 1)
scales the similarity logits to control retrieval sharpness.

This is a *memory-based* model — not recurrent, not convolutional, not
self-attentive.  It learns a codebook of representative time-series "shapes"
and uses nearest-prototype retrieval (soft, differentiable) to forecast.

Architecture comparison:
  DLinear:           global linear readout — no memory.
  TSReservoir:       fixed random echo-state reservoir.
  Transformers:      data-to-data attention over input tokens.
  PrototypicalForecaster: input-to-prototype attention; prototypes are learned.

Architecture::

    Input (B, T, C)
      → RevIN norm
      → CI reshape → (B·C, T)
      → QueryEncoder (T → d_proto, 2-layer MLP + LayerNorm)
      → prototype similarity → softmax → retrieve from K values
      → DirectMLP (T → pred_len, 2-layer MLP)
      → OutputMLP([retrieved, direct] → pred_len)
      → reshape → (B, pred_len, C)
      → RevIN denorm

Args:
    seq_len:    input lookback T.
    pred_len:   forecast horizon.
    enc_in:     number of variates C.
    n_proto:    number of prototype patterns K.
    d_proto:    prototype embedding dimension.
    query_dim:  hidden width of the query encoder MLP.
    dropout:    dropout probability.
    revin:      use RevIN instance normalisation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.revin import RevIN


class _QueryEncoder(nn.Module):
    """Encodes a scalar time series (T,) → query vector (d_proto,)."""

    def __init__(self, seq_len: int, d_proto: int, query_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, query_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim, d_proto),
            nn.LayerNorm(d_proto),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) → (B, d_proto)"""
        return self.net(x)


class PrototypicalForecaster(nn.Module):
    """Prototype memory network forecaster (channel-independent)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        n_proto: int = 32,
        d_proto: int = 64,
        query_dim: int = 128,
        dropout: float = 0.1,
        revin: bool = True,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in
        self._revin_flag = revin
        self._scale = math.sqrt(d_proto)

        if revin:
            self.revin_layer = RevIN(enc_in, affine=True)

        # Prototype memory
        self.proto_keys = nn.Parameter(torch.randn(n_proto, d_proto) * 0.02)
        self.proto_vals = nn.Parameter(torch.randn(n_proto, pred_len) * 0.02)
        # Learnable temperature for retrieval sharpness
        self.log_temp = nn.Parameter(torch.zeros(1))

        # Query encoder
        self.query_enc = _QueryEncoder(seq_len, d_proto, query_dim, dropout)

        # Direct (bypass) path
        self.direct_mlp = nn.Sequential(
            nn.Linear(seq_len, query_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim, pred_len),
        )

        # Fusion: memory + direct → prediction
        self.fusion = nn.Linear(2 * pred_len, pred_len)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, pred_len, C)"""
        B, T, C = x.shape

        if self._revin_flag:
            x = self.revin_layer(x, "norm")

        # CI: treat each channel as an independent scalar sequence
        x_ci = x.permute(0, 2, 1).reshape(B * C, T)   # (BC, T)

        # Query encoding
        query = self.query_enc(x_ci)                   # (BC, d_proto)

        # Prototype retrieval (soft nearest-neighbour)
        tau = self.log_temp.exp().clamp(min=0.1)
        sim = (query @ self.proto_keys.T) / (self._scale * tau)   # (BC, K)
        attn = F.softmax(sim, dim=-1)                              # (BC, K)
        mem_pred = attn @ self.proto_vals                          # (BC, pred_len)

        # Direct path
        dir_pred = self.direct_mlp(x_ci)               # (BC, pred_len)

        # Fusion
        fused = self.fusion(self.drop(torch.cat([mem_pred, dir_pred], dim=-1)))  # (BC, pred_len)
        pred = fused.reshape(B, C, self.pred_len).permute(0, 2, 1)

        if self._revin_flag:
            pred = self.revin_layer(pred, "denorm")
        return pred
