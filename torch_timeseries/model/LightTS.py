"""LightTS: Light Temporal Series Forecasting.

Reference: Zhang et al., "Less Is More: Fast Multivariate Time Series
Forecasting with Light Sampling-oriented MLP Structures", VLDB 2022.

Key idea:
  Instead of attending over all T time steps (expensive), LightTS uses two
  complementary **interval sampling** strategies:
    1. **Continuous sampling** — take L equally-spaced contiguous sub-sequences,
       each of length S = T / L.  This captures local temporal patterns.
    2. **Interval sampling** — downsample at stride L (every L-th step), giving
       L sub-sequences each of length S.  This captures global / periodic patterns.

  Both sampled views are processed by the same shared MLP block, then combined
  via a projection layer that maps to the target horizon.

Architecture:
  Input (B, T, N) → channel-independent reshape → two samplings:
    A. Continuous: (B*N, L, S)
    B. Interval:   (B*N, L, S)

  Each passes through a 2-layer MLP (IEBlock) → concat → linear → pred_len.

  RevIN optional for distribution shift.

Args:
    seq_len:   input lookback window T.
    pred_len:  forecasting horizon.
    enc_in:    number of variates.
    chunk_size: number of chunks L (default: min(T, 8)).
    d_model:   hidden size inside the IEBlock MLP.
    revin:     use Reversible Instance Normalisation.
    dropout:   dropout in IEBlock.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True)
            self._std = x.std(dim=1, keepdim=True) + self.eps
            x = (x - self._mean) / self._std
            x = x * self.affine_weight + self.affine_bias
        elif mode == "denorm":
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self._std + self._mean
        return x


class _IEBlock(nn.Module):
    """Interval-Enhanced MLP block shared by both sampling views."""

    def __init__(self, input_dim: int, hid_dim: int, out_dim: int,
                 num_node: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim),
        )
        # Node-wise mixing across the L dimension
        self.fc2 = nn.Sequential(
            nn.Linear(num_node, num_node),
            nn.GELU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hid_dim, out_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B*N, L, S)``  — L chunks, each of length S.

        Returns:
            ``(B*N, L, out_dim)``
        """
        # Per-chunk temporal MLP
        out = self.fc1(x)                          # (B*N, L, hid_dim)
        out = self.norm(out)

        # Cross-chunk mixing: transpose L ↔ hid_dim
        out_t = out.transpose(1, 2)                # (B*N, hid_dim, L)
        out_t = self.fc2(out_t)                    # (B*N, hid_dim, L)
        out = out_t.transpose(1, 2)                # (B*N, L, hid_dim)

        return self.fc3(out)                       # (B*N, L, out_dim)


class LightTS(nn.Module):
    """LightTS: channel-independent dual-sampling MLP forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        chunk_size: Optional[int] = None,
        d_model: int = 64,
        revin: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

        # Number of chunks L; each chunk has seg_len = T // L time steps
        L = chunk_size if chunk_size is not None else min(seq_len, 8)
        self.L = L
        self.seg_len = seq_len // L   # continuous segment length

        self.revin = revin
        if revin:
            self.revin_layer = _RevIN(enc_in)

        # Shared IE blocks for both sampling views
        self.cont_block = _IEBlock(
            input_dim=self.seg_len,
            hid_dim=d_model,
            out_dim=d_model,
            num_node=L,
            dropout=dropout,
        )
        self.intv_block = _IEBlock(
            input_dim=self.seg_len,
            hid_dim=d_model,
            out_dim=d_model,
            num_node=L,
            dropout=dropout,
        )

        # Output projection: concatenated embeddings → pred_len
        self.projection = nn.Linear(2 * d_model * L, pred_len)

    # ------------------------------------------------------------------ #
    # sampling strategies                                                  #
    # ------------------------------------------------------------------ #

    def _continuous_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Split into L contiguous chunks of length seg_len.

        Args:
            x: ``(B*N, T_padded)``

        Returns:
            ``(B*N, L, seg_len)``
        """
        BN, T = x.shape
        # Trim to L * seg_len (drop last partial chunk if any)
        T_use = self.L * self.seg_len
        x = x[:, :T_use]
        return x.reshape(BN, self.L, self.seg_len)

    def _interval_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Stride-L downsampling: group every L-th element into a chunk.

        Args:
            x: ``(B*N, T_padded)``

        Returns:
            ``(B*N, L, seg_len)``
        """
        BN, T = x.shape
        T_use = self.L * self.seg_len
        x = x[:, :T_use]
        # x[b, 0, s] = original x[b, s*L + 0]  (stride L)
        x = x.reshape(BN, self.seg_len, self.L)   # interleave by L
        return x.transpose(1, 2)                   # (BN, L, seg_len)

    # ------------------------------------------------------------------ #
    # forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, seq_len, enc_in)``.

        Returns:
            ``(B, pred_len, enc_in)``.
        """
        B, T, N = x.shape

        if self.revin:
            x = self.revin_layer(x, "norm")

        # Channel-independent: merge batch and variate dims
        x_ci = x.permute(0, 2, 1).reshape(B * N, T)   # (B*N, T)

        # Continuous sampling view
        x_cont = self._continuous_sample(x_ci)          # (B*N, L, seg_len)
        h_cont = self.cont_block(x_cont)                 # (B*N, L, d_model)

        # Interval sampling view
        x_intv = self._interval_sample(x_ci)            # (B*N, L, seg_len)
        h_intv = self.intv_block(x_intv)                 # (B*N, L, d_model)

        # Concatenate both views and project
        h = torch.cat([h_cont, h_intv], dim=-1)         # (B*N, L, 2*d_model)
        h = h.reshape(B * N, -1)                         # (B*N, 2*d_model*L)
        out = self.projection(h)                         # (B*N, pred_len)
        out = out.reshape(B, N, self.pred_len).permute(0, 2, 1)  # (B, pred_len, N)

        if self.revin:
            out = self.revin_layer(out, "denorm")

        return out
