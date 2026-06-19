"""SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion.

Reference: Han et al., "SOFTS: Efficient Multivariate Time Series Forecasting
with Series-Core Fusion", NeurIPS 2024.

Key idea:
  Standard cross-variate attention is O(N²) in the number of variates.
  SOFTS replaces all-pairs interactions with a **star topology**:
    1. Compute a single "series core" — a shared global representation of
       all variates.
    2. Each variate interacts ONLY with the core (O(N) interactions total).

  This reduces cross-variate complexity from O(N²) to O(N) while still
  allowing information to flow between all variates through the shared core.

Architecture:
  1. Input normalization (RevIN optional).
  2. Embed each variate across its lookback window: (B, T, N) → (B, N, d_model).
  3. L × STID-style SOFTS blocks:
       a. Temporal mixing via per-variate MLP.
       b. Core computation: mean-pool variates → core MLP.
       c. Core fusion: concatenate [variate, core] → linear → residual.
  4. Output projection: (B, N, d_model) → (B, N, pred_len).

Args:
    seq_len:    input lookback window.
    pred_len:   forecasting horizon.
    enc_in:     number of variates (channels).
    d_model:    per-variate representation dimension.
    d_core:     core hidden dimension (default: same as d_model).
    e_layers:   number of SOFTS blocks.
    dropout:    dropout rate in blocks.
    revin:      use Reversible Instance Normalisation (RevIN).
"""
from __future__ import annotations

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


class _SOFTSBlock(nn.Module):
    """One SOFTS block: temporal mixing + star-topology core fusion."""

    def __init__(self, d_model: int, d_core: int, dropout: float = 0.0):
        super().__init__()
        # Temporal MLP (per variate, across d_model dimension)
        self.temporal_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        # Core MLP: aggregate all variates into a shared core representation
        self.core_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_core),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_core, d_model),
        )
        # Fusion: combine each variate with the core
        self.fusion_gate = nn.Linear(2 * d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, N, d_model)``

        Returns:
            ``(B, N, d_model)``
        """
        # Temporal mixing (per-variate)
        x = x + self.drop(self.temporal_mlp(x))

        # Star-topology core fusion
        core = x.mean(dim=1, keepdim=True)          # (B, 1, d_model)
        core = self.core_mlp(core)                  # (B, 1, d_model)
        core_expand = core.expand_as(x)             # (B, N, d_model)
        fusion = self.fusion_gate(
            torch.cat([x, core_expand], dim=-1)     # (B, N, 2*d_model)
        )                                            # (B, N, d_model)
        x = x + self.drop(fusion)

        return x


class SOFTS(nn.Module):
    """SOFTS: Series-Core Fusion for efficient multivariate forecasting.

    Cross-variate interactions use a star topology (O(N)) instead of
    all-pairs attention (O(N²)).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 512,
        d_core: Optional[int] = None,
        e_layers: int = 2,
        dropout: float = 0.0,
        revin: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        d_core = d_core if d_core is not None else d_model

        self.revin = revin
        if revin:
            self.revin_layer = _RevIN(enc_in)

        # Embed lookback window per variate
        self.input_proj = nn.Linear(seq_len, d_model)

        self.blocks = nn.ModuleList([
            _SOFTSBlock(d_model, d_core, dropout) for _ in range(e_layers)
        ])

        # Project from d_model → pred_len for each variate
        self.output_proj = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, seq_len, enc_in)``.

        Returns:
            Forecast ``(B, pred_len, enc_in)``.
        """
        B, T, N = x.shape

        if self.revin:
            x = self.revin_layer(x, "norm")

        # (B, T, N) → (B, N, T) → per-variate embedding → (B, N, d_model)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        # (B, N, d_model) → (B, N, pred_len) → (B, pred_len, N)
        x = self.output_proj(x).permute(0, 2, 1)

        if self.revin:
            x = self.revin_layer(x, "denorm")

        return x
