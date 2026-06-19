"""LinearAttentionForecaster: O(n) linear self-attention for time-series.

References:
  Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive
  Transformers with Linear Attention", ICML 2020.

Key idea:
  Standard (softmax) attention is O(n²d) because every query attends to every key:

      Attn(Q, K, V) = softmax(Q K^T / √d) V

  Linear attention replaces softmax with a kernel decomposition  K(q, k) = φ(q)·φ(k)
  where φ(x) = ELU(x) + 1 (element-wise, always positive):

      LinearAttn(Q, K, V) = φ(Q) (φ(K)^T V) / φ(Q) (φ(K)^T 1)

  Re-associating the products, the numerator becomes:
      Z = φ(Q) @ (φ(K)^T @ V)   — cost O(n d²) instead of O(n² d)
      Normaliser = φ(Q) @ (φ(K)^T @ ones_d)  — same re-ordering

  This gives each token a representation that aggregates all keys/values in
  O(d²) per query, reducing the total cost from O(n² d) to O(n d²).

Architecture:
  1. RevIN normalise.
  2. Patch embed: T → n_patches tokens per variate (channel-independent).
  3. L LinearAttention layers:
       • Multi-head linear attention (ELU+1 kernel)
       • Pre-norm LayerNorm
       • SwiGLU or GELU FFN
  4. Mean-pool patches → head → pred_len.

Args:
    seq_len:    lookback length T.
    pred_len:   forecast horizon.
    enc_in:     number of variates.
    d_model:    token dimension.
    n_heads:    number of attention heads.
    e_layers:   number of LinearAttention layers.
    d_ff:       feed-forward hidden size.
    patch_len:  patch length (default 16).
    stride:     patch stride (default = patch_len).
    dropout:    dropout rate.
    revin:      use RevIN normalisation.
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
# Multi-head linear attention
# ──────────────────────────────────────────────────────────────────────────────


class _LinearMultiheadAttention(nn.Module):
    """Multi-head linear attention with ELU+1 kernel."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        """ELU(x) + 1 — strictly positive kernel feature map."""
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n, D)
        Returns:
            (B, n, D)
        """
        B, n, D = x.shape
        H, d = self.n_heads, self.d_head

        Q = self.q_proj(x).reshape(B, n, H, d).transpose(1, 2)  # (B, H, n, d)
        K = self.k_proj(x).reshape(B, n, H, d).transpose(1, 2)
        V = self.v_proj(x).reshape(B, n, H, d).transpose(1, 2)

        phi_Q = self._phi(Q)   # (B, H, n, d)
        phi_K = self._phi(K)   # (B, H, n, d)

        # Numerator: φ(Q) @ (φ(K)^T @ V)   shape (B, H, n, d)
        KtV = phi_K.transpose(-2, -1) @ V          # (B, H, d, d)
        num = phi_Q @ KtV                           # (B, H, n, d)

        # Denominator: φ(Q) @ sum_k(φ(K)[k])  shape (B, H, n, 1)
        sum_K = phi_K.sum(dim=-2, keepdim=True)     # (B, H, 1, d)
        denom = (phi_Q * sum_K).sum(dim=-1, keepdim=True).clamp(min=1e-6)

        out = num / denom                            # (B, H, n, d)
        out = out.transpose(1, 2).reshape(B, n, D)
        return self.drop(self.out_proj(out))


# ──────────────────────────────────────────────────────────────────────────────
# Transformer layer with linear attention
# ──────────────────────────────────────────────────────────────────────────────


class _LinearAttnLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.attn = _LinearMultiheadAttention(d_model, n_heads, dropout)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        h = F.gelu(self.ff1(self.norm2(x)))
        x = x + self.drop(self.ff2(h))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# LinearAttentionForecaster
# ──────────────────────────────────────────────────────────────────────────────


class LinearAttentionForecaster(nn.Module):
    """O(n) linear attention Transformer forecaster (channel-independent patches)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 128,
        patch_len: int = 16,
        stride: int | None = None,
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

        if stride is None:
            stride = patch_len
        self.patch_len = patch_len
        self.stride = stride

        n_patches = math.ceil((seq_len - patch_len) / stride) + 1
        self.n_patches = n_patches
        self.pad_len = max(0, (n_patches - 1) * stride + patch_len - seq_len)

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [_LinearAttnLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
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

        x_ci = x.transpose(1, 2).reshape(B * C, T)
        if self.pad_len > 0:
            x_ci = F.pad(x_ci, (0, self.pad_len))
        patches = x_ci.unfold(-1, self.patch_len, self.stride)  # (BC, n_patches, patch_len)

        h = self.patch_embed(patches) + self.pos_embed
        h = self.drop(h)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        pooled = h.mean(dim=1)                                  # (BC, d_model)
        out = self.head(pooled)                                 # (BC, pred_len)
        out = out.reshape(B, C, self.pred_len).transpose(1, 2)  # (B, pred_len, C)

        if self._revin_flag:
            out = self.revin_layer(out, "denorm")
        return out
