"""HDMixer: Hierarchical Dependency MLP Mixer for time-series forecasting.

Key idea:
  Time-series patterns exist at multiple temporal scales simultaneously.
  HDMixer captures them by applying a two-level MLP mixing operation at
  each of S user-specified patch sizes {p_1, ..., p_S}:

    For each scale s with patch size p_s:
      1. Reshape (B*C, T) → (B*C, n_patches, p_s)   [partition into non-overlapping patches]
      2. Intra-patch MLP  — mixes within each patch (local temporal dependencies)
      3. Inter-patch MLP  — mixes across patches (global temporal dependencies)
      4. Flatten + linear  → (B*C, d_model)          [per-scale representation]

    Aggregate: element-wise mean over all scale representations → (B*C, d_model)
    Head: Linear(d_model, pred_len) → (B*C, pred_len)

  This mirrors MLP-Mixer's row/column factorisation but applies it independently at
  multiple temporal scales, letting the model capture both fine-grained local patterns
  (small p) and long-range dependencies (large p) without explicit decomposition.

Architecture comparison:
  TSMixer:  flat single-scale patch mixing with temporal+variate MLP blocks.
  TimeMixer: seasonal-trend decomposition + past-decomp & future-mix modules.
  HDMixer:  no decomposition; multi-scale patch MLP (intra + inter) in parallel.

Args:
    seq_len:     input lookback length T.
    pred_len:    forecast horizon.
    enc_in:      number of variates.
    patch_sizes: list of patch sizes to use (default [4, 8, 16]).
    d_model:     hidden size for per-scale projection.
    dropout:     dropout rate inside MLP blocks.
    revin:       use RevIN normalisation.
"""
from __future__ import annotations

from typing import List

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
# Single-scale mixing block
# ──────────────────────────────────────────────────────────────────────────────


class _ScaleMixer(nn.Module):
    """Intra-patch + inter-patch MLP mixing at a single temporal scale."""

    def __init__(self, seq_len: int, patch_size: int, d_model: int, dropout: float):
        super().__init__()
        n_patches = (seq_len + patch_size - 1) // patch_size
        pad_len = n_patches * patch_size - seq_len
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.pad_len = pad_len

        # Intra-patch: mix within each patch over the time axis
        self.intra_norm = nn.LayerNorm(patch_size)
        self.intra_mlp = nn.Sequential(
            nn.Linear(patch_size, patch_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(patch_size * 2, patch_size),
            nn.Dropout(dropout),
        )

        # Inter-patch: mix across patches, one position at a time
        self.inter_norm = nn.LayerNorm(n_patches)
        self.inter_mlp = nn.Sequential(
            nn.Linear(n_patches, n_patches * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_patches * 2, n_patches),
            nn.Dropout(dropout),
        )

        self.proj = nn.Linear(n_patches * patch_size, d_model)

    def forward(self, x_ci: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_ci: (BC, T)
        Returns:
            (BC, d_model)
        """
        BC = x_ci.shape[0]
        if self.pad_len > 0:
            x_ci = F.pad(x_ci, (0, self.pad_len))
        x = x_ci.reshape(BC, self.n_patches, self.patch_size)  # (BC, P, p)

        # Intra-patch mixing (residual)
        x = x + self.intra_mlp(self.intra_norm(x))             # (BC, P, p)

        # Inter-patch mixing (residual) — transpose to (BC, p, P), mix over P
        x = x.transpose(1, 2)                                  # (BC, p, P)
        x = x + self.inter_mlp(self.inter_norm(x))             # (BC, p, P)
        x = x.transpose(1, 2)                                  # (BC, P, p)

        return self.proj(x.reshape(BC, -1))                    # (BC, d_model)


# ──────────────────────────────────────────────────────────────────────────────
# HDMixer
# ──────────────────────────────────────────────────────────────────────────────


class HDMixer(nn.Module):
    """Hierarchical Dependency MLP Mixer (channel-independent)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_sizes: List[int] | None = None,
        d_model: int = 64,
        dropout: float = 0.1,
        revin: bool = True,
    ):
        super().__init__()
        if patch_sizes is None:
            patch_sizes = [4, 8, 16]
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self._revin_flag = revin

        if revin:
            self.revin_layer = _RevIN(enc_in)

        self.mixers = nn.ModuleList(
            [_ScaleMixer(seq_len, p, d_model, dropout) for p in patch_sizes]
        )
        self.n_scales = len(patch_sizes)
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

        x_ci = x.transpose(1, 2).reshape(B * C, T)              # (BC, T)

        # Mix at each scale and average
        scale_outs = [mixer(x_ci) for mixer in self.mixers]     # each (BC, d_model)
        agg = torch.stack(scale_outs, dim=0).mean(dim=0)        # (BC, d_model)

        out = self.head(agg)                                     # (BC, pred_len)
        out = out.reshape(B, C, self.pred_len).transpose(1, 2)  # (B, pred_len, C)

        if self._revin_flag:
            out = self.revin_layer(out, "denorm")
        return out
