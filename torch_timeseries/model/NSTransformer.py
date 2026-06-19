"""NSTransformer: Non-stationary Transformer for time-series forecasting.

Reference: Liu et al., "Non-stationary Transformers: Exploring the Stationarity
in Time Series Forecasting", NeurIPS 2022.

Key idea:
  Vanilla Transformer applied to normalised inputs suffers from
  over-stationarization: after subtracting mean and dividing by std the
  model sees nearly identical attention scores across samples and loses the
  ability to differentiate non-stationary patterns.

  NSTransformer fixes this with two components:

    1. **Series Stationarization**: subtract per-sample temporal mean and
       divide by temporal std before the encoder (equivalent to instance
       normalization without learnable affine parameters).

    2. **De-stationary Attention**: the attention logits are rescaled by the
       *original* series statistics so that non-stationarity is re-injected
       at every layer:
            scores = tau * (Q K^T / sqrt(d_k)) + delta
       where tau (std proxy) and delta (mean proxy) are learned projections
       of the raw series statistics, one scalar pair per sample per layer.

Simplified implementation:
  - Single tau/delta projector shared across layers (lightweight).
  - Encoder-only with direct output projection (last token -> pred_len * C).
  - RevIN-style de-normalization on the output.

Args:
    seq_len:   input lookback window.
    pred_len:  forecast horizon.
    enc_in:    number of variates.
    d_model:   model dimension.
    n_heads:   attention heads.
    e_layers:  number of encoder layers.
    d_ff:      feed-forward hidden size.
    dropout:   dropout rate.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DeStationaryAttention(nn.Module):
    """Multi-head self-attention with de-stationary score rescaling."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        tau: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:     (B, T, d_model)
            tau:   (B, 1, 1) — per-sample std scaling scalar
            delta: (B, 1, 1) — per-sample mean shift scalar
        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.d_k

        q = self.q_proj(x).reshape(B, T, H, Dh).transpose(1, 2)  # (B, H, T, Dh)
        k = self.k_proj(x).reshape(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, Dh).transpose(1, 2)

        scale = math.sqrt(Dh)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T, T)

        # de-stationary rescaling: tau * scores + delta
        # tau/delta: (B, 1, 1) -> (B, 1, 1, 1) for broadcasting over H, T, T
        scores = tau.unsqueeze(1) * scores + delta.unsqueeze(1)

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)               # (B, H, T, Dh)
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)


class _NSTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = _DeStationaryAttention(d_model, n_heads, dropout)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, tau: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        # self-attention sub-layer
        x = self.norm1(x + F.dropout(self.attn(x, tau, delta), p=self.dropout, training=self.training))
        # feed-forward sub-layer
        ff = F.dropout(F.gelu(self.ff1(x)), p=self.dropout, training=self.training)
        ff = F.dropout(self.ff2(ff), p=self.dropout, training=self.training)
        return self.norm2(x + ff)


class NSTransformer(nn.Module):
    """Non-stationary Transformer.

    Encoder-only architecture.  Input is stationarised; original statistics
    are re-injected via de-stationary attention at each layer.  Output is
    produced from the last token and de-normalised before returning.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 256,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

        # tau/delta projectors: scalar statistics -> attention-space scalars
        # input: (B, 1) — mean/std averaged over variates; output: (B, 1)
        self.tau_proj = nn.Sequential(
            nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1)
        )
        self.delta_proj = nn.Sequential(
            nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1)
        )

        self.input_proj = nn.Linear(enc_in, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList(
            [_NSTransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, pred_len * enc_in)
        self.dropout = dropout

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)
        Returns:
            (B, pred_len, C)
        """
        B, T, C = x.shape

        # 1. series stationarisation
        mu = x.mean(dim=1, keepdim=True)           # (B, 1, C)
        sigma = x.std(dim=1, keepdim=True).clamp(min=1e-5)  # (B, 1, C)
        x_norm = (x - mu) / sigma

        # 2. compute tau, delta for de-stationary attention
        #    collapse variates dimension -> (B, 1, 1)
        tau_in = sigma.mean(dim=-1)     # (B, 1)
        delta_in = mu.mean(dim=-1)      # (B, 1)
        tau = self.tau_proj(tau_in).unsqueeze(-1)    # (B, 1, 1)
        delta = self.delta_proj(delta_in).unsqueeze(-1)  # (B, 1, 1)

        # 3. embed + positional
        h = self.input_proj(x_norm) + self.pos_embed[:, :T, :]
        h = F.dropout(h, p=self.dropout, training=self.training)

        # 4. encoder layers
        for layer in self.layers:
            h = layer(h, tau, delta)
        h = self.norm(h)

        # 5. decode from last token
        out = self.output_proj(h[:, -1, :])         # (B, pred_len * C)
        out = out.reshape(B, self.pred_len, C)

        # 6. de-normalise
        out = out * sigma + mu                       # broadcast (B, 1, C) over pred_len
        return out
