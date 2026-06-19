"""GATForecaster: Graph Attention Network for multivariate time-series forecasting.

References:
  Veličković et al., "Graph Attention Networks", ICLR 2018.

Key idea:
  Treat each variate as a graph node.  Instead of applying the same weight to all
  neighbour aggregations (like spectral GCN), GAT computes a per-edge attention
  weight α_{ij} and takes a weighted sum of neighbour features:

      e_{ij}  = LeakyReLU( a^T [ W h_i || W h_j ] )     (raw attention score)
      α_{ij}  = softmax_j( e_{ij} + b_{ij} )              (normalised + edge bias)
      h_i'    = σ( Σ_j  α_{ij}  W_V  h_j )               (weighted update)

  The learnable *static edge bias* b_{ij} ∈ ℝ^{n_heads × C × C} lets the model
  encode dataset-specific inter-variate relationships that persist across all time
  steps (similar to positional bias in Graphformer, ICLR 2022).

  Architecture comparison:
    iTransformer:   standard softmax self-attention on C variate tokens; no edge bias.
    GCNForecaster:  fixed A = softmax(ReLU(E1·E2ᵀ)), polynomial graph conv + TCN.
    GATForecaster:  content-based edge attention α_{ij} + learnable edge bias b_{ij}.

Architecture:
  1. RevIN normalise.
  2. Temporal encoder: Linear(T → d_model) per variate → (B, C, d_model).
  3. L GAT layers (multi-head GAT + FFN + pre-norm).
  4. Forecast head: Linear(d_model → pred_len) per variate → (B, C, pred_len)
     → transpose → (B, pred_len, C).

Args:
    seq_len:     input lookback T.
    pred_len:    forecast horizon.
    enc_in:      number of variates C (= number of graph nodes).
    d_model:     node feature dimension.
    n_heads:     number of attention heads.
    e_layers:    number of GAT layers.
    d_ff:        feed-forward hidden size.
    dropout:     dropout rate.
    revin:       use RevIN normalisation.
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
# Multi-head Graph Attention layer
# ──────────────────────────────────────────────────────────────────────────────


class _GATLayer(nn.Module):
    """Multi-head GAT with learnable static edge bias."""

    def __init__(self, d_model: int, n_heads: int, n_nodes: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        # Learnable static edge bias: (n_heads, C, C)
        self.edge_bias = nn.Parameter(torch.zeros(n_heads, n_nodes, n_nodes))
        nn.init.normal_(self.edge_bias, std=0.02)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, C, d_model)
        Returns:
            (B, C, d_model)
        """
        B, C, D = h.shape
        H, d = self.n_heads, self.d_head

        Q = self.q_proj(h).reshape(B, C, H, d).transpose(1, 2)  # (B, H, C, d)
        K = self.k_proj(h).reshape(B, C, H, d).transpose(1, 2)
        V = self.v_proj(h).reshape(B, C, H, d).transpose(1, 2)

        # Scaled dot-product scores
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d)        # (B, H, C, C)

        # Add learnable static edge bias
        scores = scores + self.edge_bias.unsqueeze(0)             # (B, H, C, C)

        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, C, D)        # (B, C, D)
        return self.out_proj(out)


# ──────────────────────────────────────────────────────────────────────────────
# GAT Transformer-style layer
# ──────────────────────────────────────────────────────────────────────────────


class _GATTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_nodes: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gat = _GATLayer(d_model, n_heads, n_nodes, dropout)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h + self.gat(self.norm1(h))
        h = h + self.drop(self.ff2(F.gelu(self.ff1(self.norm2(h)))))
        return h


# ──────────────────────────────────────────────────────────────────────────────
# GATForecaster
# ──────────────────────────────────────────────────────────────────────────────


class GATForecaster(nn.Module):
    """Graph Attention Network forecaster (channel-mixing, variate graph)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 128,
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

        # Temporal encoder: embed each variate's time series into d_model
        self.embed = nn.Linear(seq_len, d_model)

        # GAT layers
        self.layers = nn.ModuleList(
            [_GATTransformerLayer(d_model, n_heads, enc_in, d_ff, dropout)
             for _ in range(e_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

        # Per-variate forecast head
        self.head = nn.Linear(d_model, pred_len)

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

        # Embed each variate: (B, C, T) → (B, C, d_model)
        h = self.embed(x.transpose(1, 2))

        # GAT encoding on variate graph
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)                                          # (B, C, d_model)

        out = self.head(h)                                        # (B, C, pred_len)
        out = out.transpose(1, 2)                                 # (B, pred_len, C)

        if self._revin_flag:
            out = self.revin_layer(out, "denorm")
        return out
