"""RetForecaster: Retentive Network for time-series forecasting.

Reference: Sun et al., "Retentive Network: A Successor to Transformer for
Large Language Models", arXiv 2307.08621, 2023.

Key ideas:
  Standard self-attention computes softmax(QK^T / √d) V.  RetNet replaces
  this with a **retention** mechanism that uses an explicit **exponential
  causal decay** mask instead of softmax:

      Ret(Q, K, V) = Q (K^T ⊙ D) V
      D[m, n] = γ^(m-n)  if m ≥ n  else  0          (causal, lower-triangular)

  where γ ∈ (0, 1) is a per-head learnable decay rate.

  Key properties:
    • **Parallel training**: like Transformer — compute all positions at once.
    • **Recurrent inference**: like RNN — O(1) state update per step.
    • **No softmax**: the normalisation is done via a denominator
      D_norm[m] = Σ_n D[m,n] (= Σ_{k=0}^{m-1} γ^k).
    • **Multi-scale retention (MSR)**: different heads use different γ values,
      enabling multi-resolution temporal memory.

  For time-series forecasting we apply this channel-independently with
  patch tokenisation (identical to PatchTST), substituting retention for
  attention in each Transformer layer.

  Architecture:
    1. RevIN normalise.
    2. Patch embed: T → n_patches tokens per variate.
    3. L RetNet layers (Multi-Scale Retention + SwiGLU FFN + LayerNorm).
    4. Mean-pool patches → head → pred_len.

Args:
    seq_len:     input lookback length.
    pred_len:    forecast horizon.
    enc_in:      number of variates.
    d_model:     token dimension.
    n_heads:     number of retention heads.
    e_layers:    number of RetNet layers.
    d_ff:        feed-forward hidden size.
    patch_len:   patch length.
    stride:      patch stride (default = patch_len).
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
# Multi-Scale Retention
# ──────────────────────────────────────────────────────────────────────────────


class _MultiScaleRetention(nn.Module):
    """Retention with learnable per-head decay rates γ_h ∈ (0, 1).

    Parallel form:
        Q = x W_Q,  K = x W_K,  V = x W_V   (each head-split)
        D[m,n] = γ^(m-n) if m>=n else 0
        Norm[m] = max(|Σ_n D[m,n] * Q_m · K_n|, 1)  — absolute value gating
        Ret = D ⊙ (Q K^T) / Norm  @  V
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Decay rates: γ_h = 1 - 2^{-(5 + h)}  (from RetNet paper defaults)
        gammas = 1.0 - 2.0 ** (
            -5.0 - torch.arange(n_heads, dtype=torch.float32)
        )
        self.gamma = nn.Parameter(gammas)  # learnable

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.gn = nn.GroupNorm(n_heads, d_model)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def _causal_decay_mask(self, L: int, gamma: torch.Tensor) -> torch.Tensor:
        """Build (n_heads, L, L) causal decay mask.

        D[h, m, n] = gamma[h]^(m-n) if m>=n else 0
        """
        idx = torch.arange(L, device=gamma.device, dtype=gamma.dtype)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)   # (L, L)  diff[m,n] = m-n
        # Lower-triangular (m >= n): mask out upper triangle
        mask = torch.where(diff >= 0, gamma.view(-1, 1, 1) ** diff.unsqueeze(0),
                           torch.zeros_like(diff.unsqueeze(0).expand(len(gamma), -1, -1)))
        return mask                                   # (n_heads, L, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        B, L, D = x.shape
        H, d = self.n_heads, self.d_head

        Q = self.W_Q(x).reshape(B, L, H, d).transpose(1, 2)  # (B, H, L, d)
        K = self.W_K(x).reshape(B, L, H, d).transpose(1, 2)
        V = self.W_V(x).reshape(B, L, H, d).transpose(1, 2)

        # Decay mask: (H, L, L)
        gamma = torch.sigmoid(self.gamma)             # ensure (0,1)
        D_mask = self._causal_decay_mask(L, gamma).to(x.dtype)

        # Retention scores: (B, H, L, L)
        scores = (Q @ K.transpose(-2, -1)) * D_mask.unsqueeze(0)

        # Normalise by row-sum magnitude (gating as in paper)
        row_sum = scores.abs().sum(-1, keepdim=True).clamp(min=1.0)
        ret = (scores / row_sum) @ V                  # (B, H, L, d)

        ret = ret.transpose(1, 2).reshape(B, L, D)
        ret = self.gn(ret.reshape(B * L, D)).reshape(B, L, D)
        gate = F.silu(self.gate(x))
        return self.drop(self.proj(ret * gate))


# ──────────────────────────────────────────────────────────────────────────────
# RetNet layer
# ──────────────────────────────────────────────────────────────────────────────


class _RetNetLayer(nn.Module):
    """One RetNet layer: MSR + SwiGLU FFN + pre-norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.msr = _MultiScaleRetention(d_model, n_heads, dropout)
        # SwiGLU FFN: two parallel linear layers, one gated with SiLU
        self.ffn_gate = nn.Linear(d_model, d_ff)
        self.ffn_val = nn.Linear(d_model, d_ff)
        self.ffn_out = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.msr(self.norm1(x))
        h = F.silu(self.ffn_gate(self.norm2(x))) * self.ffn_val(self.norm2(x))
        x = x + self.drop(self.ffn_out(h))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# RetForecaster
# ──────────────────────────────────────────────────────────────────────────────


class RetForecaster(nn.Module):
    """Retentive Network forecaster (channel-independent patches)."""

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
            [_RetNetLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
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
        patches = x_ci.unfold(-1, self.patch_len, self.stride)  # (B*C, n_patches, patch_len)

        h = self.patch_embed(patches) + self.pos_embed
        h = self.drop(h)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        pooled = h.mean(dim=1)                                   # (B*C, d_model)
        out = self.head(pooled)                                  # (B*C, pred_len)
        out = out.reshape(B, C, self.pred_len).transpose(1, 2)  # (B, pred_len, C)

        if self._revin_flag:
            out = self.revin_layer(out, "denorm")
        return out
