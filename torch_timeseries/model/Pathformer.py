"""Pathformer: Multi-scale Patch Transformer with Adaptive Path Selection.

Reference: Chen et al., "Pathformer: Multi-scale Transformers with Adaptive
Pathways for Time Series Forecasting", ICLR 2024.

Key ideas:
  Real-world time series contain temporal patterns at multiple granularities
  simultaneously: short-term fluctuations, daily cycles, weekly trends, etc.
  A fixed patch size forces the model to commit to a single temporal resolution.
  Pathformer addresses this via:

  1. **Multi-scale Patching**: The lookback window is divided into patches at K
     different scales (patch sizes p_1 < p_2 < ... < p_K).  Smaller patches
     capture fine-grained local patterns; larger patches capture coarse global
     trends.  Each scale produces its own sequence of n_k patch tokens.

  2. **Scale-wise Transformer Encoders**: Each scale has its own lightweight
     Transformer that applies self-attention over the n_k tokens of that scale.
     All scales share model dimension d_model but process independently.

  3. **Adaptive Path Router**: A gating network (MLP over the average-pooled
     token sequence of each scale) produces K non-negative scalar weights,
     normalised by softmax.  The final prediction is the weighted sum of
     per-scale predictions:
         pred = Σ_k  router_k(x) * pred_k(x)
     The router is input-dependent, allowing the model to lean on coarser or
     finer scales depending on the instance.

  4. **RevIN** normalisation.

Pipeline:
    x → RevIN → {for each scale k: Patch_k → PatchEmbed_k → Transformer_k
                  → pool → Predict_k}  →  Router  →  weighted blend  →  denorm

Args:
    seq_len:      input lookback length.
    pred_len:     forecast horizon.
    enc_in:       number of variates.
    patch_sizes:  list of patch sizes (ascending), e.g. [4, 8, 16].
    d_model:      hidden dimension (shared across scales).
    n_heads:      attention heads (shared across scales).
    e_layers:     number of Transformer layers per scale.
    d_ff:         feed-forward hidden size.
    dropout:      dropout rate.
    revin:        use RevIN instance normalisation.
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Shared building blocks
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


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.reshape(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.reshape(B, L, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = self.drop(torch.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class _TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.attn = _MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Per-scale branch
# ──────────────────────────────────────────────────────────────────────────────


class _ScaleBranch(nn.Module):
    """One scale of the Pathformer: patchify → Transformer → predict."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_size: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.patch_size = patch_size
        stride = patch_size  # non-overlapping patches at each scale
        n_patches = math.ceil(seq_len / patch_size)
        self.n_patches = n_patches

        # Pad seq_len to be divisible by patch_size
        self.pad_len = n_patches * patch_size - seq_len

        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList(
            [_TransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # Per-scale head: mean-pooled d_model → pred_len
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x_ci: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_ci: (B*C, T)
        Returns:
            pred:   (B*C, pred_len)  per-scale prediction
            pooled: (B*C, d_model)   mean-pooled token representation (for router)
        """
        BC, T = x_ci.shape
        if self.pad_len > 0:
            x_ci = F.pad(x_ci, (0, self.pad_len))                  # (BC, n*p)

        # Reshape to patches
        h = x_ci.reshape(BC, self.n_patches, self.patch_size)       # (BC, n_patches, p)
        h = self.patch_embed(h) + self.pos_embed                    # (BC, n_patches, d_model)
        h = self.drop(h)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)                                            # (BC, n_patches, d_model)

        pooled = h.mean(dim=1)                                      # (BC, d_model)
        pred = self.head(pooled)                                    # (BC, pred_len)
        return pred, pooled


# ──────────────────────────────────────────────────────────────────────────────
# Pathformer
# ──────────────────────────────────────────────────────────────────────────────


class Pathformer(nn.Module):
    """Multi-scale Transformer with Adaptive Path Selection."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_sizes: List[int] | None = None,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        revin: bool = True,
    ):
        super().__init__()
        if patch_sizes is None:
            patch_sizes = [4, 8, 16]
        assert len(patch_sizes) >= 1, "Need at least one patch size"

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_sizes = patch_sizes
        self.n_scales = len(patch_sizes)
        self._revin_flag = revin

        if revin:
            self.revin_layer = _RevIN(enc_in)

        # Scale-specific branches (channel-independent)
        self.branches = nn.ModuleList([
            _ScaleBranch(seq_len, pred_len, ps, d_model, n_heads, e_layers, d_ff, dropout)
            for ps in patch_sizes
        ])

        # Adaptive router: concatenated pooled representations → K weights
        # Input: n_scales * d_model per (B*C) position
        self.router = nn.Sequential(
            nn.Linear(self.n_scales * d_model, self.n_scales * 2),
            nn.GELU(),
            nn.Linear(self.n_scales * 2, self.n_scales),
        )

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

        # Channel-independent: (B, T, C) → (B*C, T)
        x_ci = x.transpose(1, 2).reshape(B * C, T)

        preds = []
        pooled_list = []
        for branch in self.branches:
            pred_k, pooled_k = branch(x_ci)  # (B*C, pred_len), (B*C, d_model)
            preds.append(pred_k)
            pooled_list.append(pooled_k)

        # Router: (B*C, K*d_model) → (B*C, K) softmax weights
        pooled_cat = torch.cat(pooled_list, dim=-1)               # (B*C, K*d_model)
        weights = torch.softmax(self.router(pooled_cat), dim=-1)  # (B*C, K)

        # Weighted blend: Σ_k w_k * pred_k → (B*C, pred_len)
        preds_stack = torch.stack(preds, dim=-1)                   # (B*C, pred_len, K)
        out = (preds_stack * weights.unsqueeze(1)).sum(-1)         # (B*C, pred_len)

        # Reshape: (B*C, pred_len) → (B, C, pred_len) → (B, pred_len, C)
        out = out.reshape(B, C, self.pred_len).transpose(1, 2)

        if self._revin_flag:
            out = self.revin_layer(out, "denorm")
        return out
