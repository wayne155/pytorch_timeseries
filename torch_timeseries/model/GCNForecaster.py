"""GCNForecaster: Adaptive Graph Convolutional Network for multivariate forecasting.

Inspired by: Wu et al., "Connecting the Dots: Multivariate Time Series
Forecasting with Graph Neural Networks", KDD 2020 (Graph WaveNet / MTGNN).

Key ideas:
  Most time-series models treat channels as either fully independent (CI)
  or fully mixed via attention/linear.  GNNs offer a middle ground: they
  model *pairwise* channel relationships explicitly through an adjacency
  matrix, and propagate information only along edges.

  1. **Learnable Adaptive Adjacency**:
     Two node embedding matrices E1, E2 ∈ R^{C × d_emb} are learned.  The
     adjacency is computed as:
         A = softmax(ReLU(E1 @ E2^T))
     This gives a directed weighted graph that is fully data-driven (no need
     for pre-computed correlation matrices) and efficiently differentiable.

  2. **Graph Convolution**:
     One GCN step: H' = σ(A_hat @ H @ W_gcn)
     where A_hat = A + I (self-loops).  We apply L hop aggregation
     (mix-hop style): H_out = Σ_{k=0..K_hops} alpha_k * A^k @ H @ W_k
     with learnable mixing weights alpha.

  3. **Temporal Processing**:
     A lightweight 1-D dilated convolution (dilations 1, 2, 4, ...) captures
     temporal patterns within each variate.  This is applied interleaved with
     graph convolution layers.

  4. **RevIN** normalisation.

Pipeline:
    x → RevIN → [TemporalConv → GraphConv] × e_layers
              → flatten → Linear → pred_len → denorm

Args:
    seq_len:       input lookback length.
    pred_len:      forecast horizon.
    enc_in:        number of variates (graph nodes).
    d_model:       feature dimension at each layer.
    e_layers:      number of (TemporalConv + GCN) layer pairs.
    d_emb:         node embedding dimension for adaptive adjacency.
    k_hops:        number of propagation hops per GCN layer.
    kernel_size:   temporal conv kernel size.
    dropout:       dropout rate.
    revin:         use RevIN normalisation.
"""
from __future__ import annotations

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
# Adaptive Adjacency
# ──────────────────────────────────────────────────────────────────────────────


class _AdaptiveAdjacency(nn.Module):
    """Learnable directed adjacency: A = softmax(ReLU(E1 @ E2^T))."""

    def __init__(self, n_nodes: int, d_emb: int):
        super().__init__()
        self.E1 = nn.Parameter(torch.empty(n_nodes, d_emb))
        self.E2 = nn.Parameter(torch.empty(n_nodes, d_emb))
        nn.init.xavier_uniform_(self.E1)
        nn.init.xavier_uniform_(self.E2)

    def forward(self) -> torch.Tensor:
        """Returns: (n_nodes, n_nodes) adjacency matrix (row-normalised)."""
        logits = F.relu(self.E1 @ self.E2.T)
        return F.softmax(logits, dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-hop Graph Convolution
# ──────────────────────────────────────────────────────────────────────────────


class _GraphConvLayer(nn.Module):
    """Multi-hop GCN with learnable per-hop mixing weights."""

    def __init__(self, d_in: int, d_out: int, k_hops: int, dropout: float):
        super().__init__()
        self.k_hops = k_hops
        # One weight matrix per hop (0-hop = self, 1..K = neighbours)
        self.weights = nn.ModuleList(
            [nn.Linear(d_in, d_out, bias=False) for _ in range(k_hops + 1)]
        )
        self.alpha = nn.Parameter(torch.ones(k_hops + 1) / (k_hops + 1))
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, C, d_in)   node features
            A: (C, C)          adjacency (row-normalised)
        Returns:
            (B, C, d_out)
        """
        alpha = torch.softmax(self.alpha, dim=0)
        out = torch.zeros(h.shape[0], h.shape[1], self.weights[0].out_features,
                          device=h.device, dtype=h.dtype)
        h_pow = h
        for k in range(self.k_hops + 1):
            out = out + alpha[k] * self.weights[k](h_pow)
            if k < self.k_hops:
                # A @ h_pow: (C,C) @ (B,C,d) — einsum for batch
                h_pow = torch.einsum("cn,bnd->bcd", A, h_pow)
        return self.drop(self.act(self.norm(out)))


# ──────────────────────────────────────────────────────────────────────────────
# Temporal Conv
# ──────────────────────────────────────────────────────────────────────────────


class _TemporalBlock(nn.Module):
    """Dilated causal Conv1d + residual + LayerNorm."""

    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(d_model, d_model, kernel_size,
                              padding=pad, dilation=dilation, groups=d_model)
        self.pw = nn.Conv1d(d_model, d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self._pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*C, d_model, T)
        Returns:
            (B*C, d_model, T)
        """
        h = self.conv(x)
        if self._pad > 0:
            h = h[..., :-self._pad]          # causal: remove future
        h = self.act(self.pw(h))
        return self.drop(h) + x              # residual


# ──────────────────────────────────────────────────────────────────────────────
# GCNForecaster
# ──────────────────────────────────────────────────────────────────────────────


class GCNForecaster(nn.Module):
    """Adaptive GCN + dilated temporal convolution for multivariate forecasting."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        e_layers: int = 3,
        d_emb: int = 10,
        k_hops: int = 2,
        kernel_size: int = 3,
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

        # Learnable adjacency
        self.adj = _AdaptiveAdjacency(enc_in, d_emb)

        # Input projection: 1 → d_model per variate
        self.input_proj = nn.Linear(1, d_model)

        # Interleaved temporal + graph conv layers
        self.temp_blocks = nn.ModuleList()
        self.gcn_layers = nn.ModuleList()
        for i in range(e_layers):
            dilation = 2 ** i
            self.temp_blocks.append(
                _TemporalBlock(d_model, kernel_size, dilation, dropout)
            )
            self.gcn_layers.append(
                _GraphConvLayer(d_model, d_model, k_hops, dropout)
            )

        # Output: flatten (d_model * T) → pred_len per variate
        self.output_proj = nn.Linear(d_model * seq_len, pred_len)

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

        # Input proj: (B, T, C) → (B, C, T, 1) → (B, C, T, d_model)
        x_var = x.transpose(1, 2)                          # (B, C, T)
        h = self.input_proj(x_var.unsqueeze(-1))           # (B, C, T, d_model)
        h = h.transpose(2, 3)                              # (B, C, d_model, T)

        # Compute adjacency once
        A = self.adj()                                     # (C, C)

        # Interleaved temporal + graph conv
        for temp, gcn in zip(self.temp_blocks, self.gcn_layers):
            # Temporal: channel-independent (B*C, d_model, T)
            BC = B * C
            h_flat = h.reshape(BC, h.shape[2], T)
            h_flat = temp(h_flat)                          # (B*C, d_model, T)
            h = h_flat.reshape(B, C, h.shape[2], T)

            # Graph: across channels (B, C, d_model)
            # Mean-pool time for graph message passing
            h_pool = h.mean(-1)                            # (B, C, d_model)
            h_pool = gcn(h_pool, A)                        # (B, C, d_model)
            # Add graph features back to all time positions
            h = h + h_pool.unsqueeze(-1)                   # (B, C, d_model, T)

        # Output: flatten d_model×T → pred_len
        h = h.reshape(B, C, -1)                            # (B, C, d_model*T)
        out = self.output_proj(h)                          # (B, C, pred_len)
        out = out.transpose(1, 2)                          # (B, pred_len, C)

        if self._revin_flag:
            out = self.revin_layer(out, "denorm")
        return out
