"""RLinear: Reversible Linear model for time-series forecasting.

Reference: Li et al., "Revisiting Long-term Time Series Forecasting: An
Investigation on Linear Mapping", arXiv 2023.

Key ideas:
  Several works showed that simple linear models can be surprisingly competitive
  with complex transformer-based methods when combined with proper normalisation.

  RLinear is the simplest possible design:
    1. **RevIN normalisation**: per-sample, per-channel instance normalisation
       with learnable affine parameters.  Removes distribution shift between
       training and test windows.
    2. **Individual Linear layer** (optional): one independent linear mapping
       per variate (T → pred_len), analogous to DLinear's "individual" mode.
       When `individual=False`, a single shared linear is used for all variates.
    3. **RevIN de-normalisation**: reverse the normalisation on the output.

  This is essentially DLinear (without decomposition) + RevIN normalisation.
  The key insight is that RevIN + Linear can match or exceed many complex models
  on benchmark datasets, demonstrating that normalisation is often the bottleneck.

Args:
    seq_len:    input lookback window.
    pred_len:   forecast horizon.
    enc_in:     number of variates.
    individual: use per-variate linear (True) or shared linear (False).
"""
from __future__ import annotations

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
            self._mean = x.mean(1, keepdim=True)
            self._std = x.std(1, keepdim=True).clamp(self.eps)
            x = (x - self._mean) / self._std
            return x * self.affine_weight + self.affine_bias
        else:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            return x * self._std + self._mean


class RLinear(nn.Module):
    """Reversible Linear: RevIN normalisation + per-channel (or shared) linear mapping.

    The simplest possible forecasting model that leverages instance normalisation
    to handle distribution shift.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        individual: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.individual = individual

        self.revin_layer = _RevIN(enc_in)

        if individual:
            # One linear per variate: (T,) → (pred_len,)
            self.linears = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(enc_in)]
            )
        else:
            # Single shared linear
            self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)
        Returns:
            (B, pred_len, C)
        """
        # 1. RevIN normalisation
        x = self.revin_layer(x, "norm")

        # 2. Linear mapping: (B, T, C) → (B, pred_len, C)
        if self.individual:
            # Apply one linear per variate
            out = torch.stack(
                [self.linears[i](x[:, :, i]) for i in range(self.enc_in)],
                dim=-1,
            )  # (B, pred_len, C)
        else:
            # Shared linear over time dimension: (B, C, T) → (B, C, pred_len)
            out = self.linear(x.transpose(1, 2)).transpose(1, 2)

        # 3. RevIN de-normalisation
        out = self.revin_layer(out, "denorm")
        return out
