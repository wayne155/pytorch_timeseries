"""Basisformer: Learnable Basis Functions for time-series forecasting.

Reference: Ni et al., "Basisformer: Attention-based Time Series Forecasting
with Learnable and Interpretable Basis", NeurIPS 2023.

Key ideas:
  Many time series decomposition methods (Fourier, Legendre, wavelets) rely on
  fixed analytical basis functions.  Basisformer learns a compact set of K
  basis functions directly from data, allowing the model to discover the most
  task-relevant decomposition.

  1. **Learnable Basis Bank**:
     A set of K prototypical time series B ∈ R^{K × T} are stored as
     parameters and learned end-to-end.  They act as a dictionary of
     "temporal atoms" — think of them as learnable sine/cosine-like patterns.

  2. **Soft Basis Assignment**:
     Each variate x_c ∈ R^T is projected to a K-dimensional coefficient vector
     via cross-attention between the variate and the K basis:
         coeff_c = softmax(x_c B^T / √T) ∈ R^K
     This gives a probability-like assignment — how much of each basis is
     present in variate c.

  3. **Coefficient Transformer**:
     The coefficient matrix Coeff ∈ R^{C × K} is treated as a sequence of K
     "basis tokens" (each of length C) or C "variate tokens" (each of length K).
     We use variate-wise: each variate is represented as a K-dim vector, and
     we apply self-attention across the K-dim axis (channel-independent).

  4. **Prediction via Basis Reconstruction**:
     The enriched coefficients α̂ ∈ R^{C × K} are used to predict:
         pred_c = α̂_c · B_pred  where B_pred ∈ R^{K × pred_len} is a
     learnable prediction basis (second set of K basis functions in R^{pred_len}).
     This gives pred_c = Σ_k α̂_{c,k} * B_pred[k, :].

  5. **RevIN** normalisation.

  Pipeline:
    x → RevIN → CrossAttn(x, B) → coeff  → SelfAttn over K dim
              → coeff_enriched  → coeff @ B_pred  → pred → denorm

Args:
    seq_len:    input lookback length.
    pred_len:   forecast horizon.
    enc_in:     number of variates.
    n_basis:    number of learnable basis functions K.
    d_model:    hidden dimension for the coefficient transformer.
    n_heads:    attention heads.
    e_layers:   number of transformer layers on coefficients.
    d_ff:       feed-forward hidden size.
    dropout:    dropout rate.
    revin:      use RevIN instance normalisation.
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
# Coefficient Transformer
# ──────────────────────────────────────────────────────────────────────────────


class _CoeffTransformerLayer(nn.Module):
    """Transformer layer operating on the K-dim coefficient space."""

    def __init__(self, n_basis: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.proj_in = nn.Linear(n_basis, d_model)
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj_out_attn = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, n_basis)
        self.drop = nn.Dropout(dropout)

    def forward(self, coeff: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coeff: (B, C, K)
        Returns:
            (B, C, K)
        """
        B, C, K = coeff.shape
        # Project K → d_model
        h = self.proj_in(coeff)                                    # (B, C, d_model)
        # Self-attention over the C variates (treat C as sequence)
        h = self.norm1(h)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.reshape(B, C, self.n_heads, self.d_head).transpose(1, 2)
        k = k.reshape(B, C, self.n_heads, self.d_head).transpose(1, 2)
        v = v.reshape(B, C, self.n_heads, self.d_head).transpose(1, 2)
        attn = self.drop(torch.softmax(
            q @ k.transpose(-2, -1) / math.sqrt(self.d_head), dim=-1))
        h_attn = (attn @ v).transpose(1, 2).reshape(B, C, -1)
        h = h + self.proj_out_attn(h_attn)
        h = h + self.ff(self.norm2(h))
        return coeff + self.proj_out(h)                            # residual on K-space


# ──────────────────────────────────────────────────────────────────────────────
# Basisformer
# ──────────────────────────────────────────────────────────────────────────────


class Basisformer(nn.Module):
    """Learnable basis bank with cross-attention decomposition."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        n_basis: int = 32,
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
        self.n_basis = n_basis
        self._revin_flag = revin

        if revin:
            self.revin_layer = _RevIN(enc_in)

        # Learnable lookup basis: K × T (each row = one basis function over T)
        self.basis_in = nn.Parameter(torch.empty(n_basis, seq_len))
        nn.init.orthogonal_(self.basis_in)

        # Learnable prediction basis: K × pred_len
        self.basis_pred = nn.Parameter(torch.empty(n_basis, pred_len))
        nn.init.orthogonal_(self.basis_pred)

        # Transformer on coefficient space
        self.coeff_layers = nn.ModuleList([
            _CoeffTransformerLayer(n_basis, d_model, n_heads, d_ff, dropout)
            for _ in range(e_layers)
        ])

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

        # x_var: (B, C, T)
        x_var = x.transpose(1, 2)

        # Cross-attention: each variate attends over the K basis functions
        # score[c, k] = x_var[c, :] · basis_in[k, :] / sqrt(T)
        # (B, C, T) @ (T, K) → (B, C, K)
        scores = x_var @ self.basis_in.T / math.sqrt(T)
        coeff = torch.softmax(scores, dim=-1)                      # (B, C, K)
        coeff = self.drop(coeff)

        # Enrich coefficients with cross-variate attention
        for layer in self.coeff_layers:
            coeff = layer(coeff)                                   # (B, C, K)

        # Reconstruct prediction: (B, C, K) @ (K, pred_len) → (B, C, pred_len)
        out = coeff @ self.basis_pred                              # (B, C, pred_len)
        out = out.transpose(1, 2)                                  # (B, pred_len, C)

        if self._revin_flag:
            out = self.revin_layer(out, "denorm")
        return out
