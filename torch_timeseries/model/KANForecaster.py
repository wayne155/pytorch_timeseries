"""KANForecaster: Kolmogorov-Arnold Network for time-series forecasting.

Reference: Liu et al., "KAN: Kolmogorov-Arnold Networks", arXiv 2404.19756, 2024.

Key idea:
  Standard MLPs compute: y_j = σ( Σ_i w_{ij} x_i )
  KANs instead compute:  y_j = Σ_i φ_{ij}(x_i)

  where each φ_{ij} is a *learnable univariate function* rather than a fixed
  activation applied to a weighted sum.  This preserves the universal
  approximation theorem while enabling more expressive activation landscapes.

  We parameterise each φ with a truncated Chebyshev polynomial basis:

      φ_{ij}(x) = Σ_{k=0}^{K} c_{ijk} · T_k(x / ‖x‖_∞)

  where T_k are Chebyshev polynomials of the first kind:
      T_0(x) = 1,  T_1(x) = x,  T_{k+1}(x) = 2x T_k(x) − T_{k-1}(x)

  All c_{ijk} are learnable parameters; the Chebyshev recurrence is evaluated
  in the forward pass — no precomputed grids required.

Architecture:
  1. RevIN normalise.
  2. Channel-independent: (B, T, C) → (B*C, T).
  3. Normalise inputs to [−1, 1] per sample (required by Chebyshev domain).
  4. L KAN layers (ChebyKANLayer): (B*C, T) → (B*C, hidden) → ... → (B*C, pred_len).
  5. RevIN denormalise.

Args:
    seq_len:   input lookback T.
    pred_len:  forecast horizon.
    enc_in:    number of variates.
    hidden:    hidden dimension between KAN layers.
    e_layers:  number of KAN layers (excluding head).
    degree:    Chebyshev polynomial degree K.
    dropout:   dropout applied between layers.
    revin:     use RevIN normalisation.
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
# Chebyshev KAN layer
# ──────────────────────────────────────────────────────────────────────────────


class _ChebyKANLayer(nn.Module):
    """One KAN layer using Chebyshev polynomial univariate functions.

    y_j = Σ_i  Σ_{k=0}^{degree} c_{ijk} · T_k(x̃_i)

    where x̃ = clamp(x / (|x|.max(-1) + ε), -1, 1).
    """

    def __init__(self, in_features: int, out_features: int, degree: int = 5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        # Chebyshev coefficients: shape (out_features, in_features, degree+1)
        self.coeffs = nn.Parameter(
            torch.empty(out_features, in_features, degree + 1)
        )
        nn.init.normal_(self.coeffs, mean=0.0, std=0.1 / (degree + 1))

    def _chebyshev_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate T_0..T_K at each element of x.

        Args:
            x: (B, in_features) — values in [−1, 1]
        Returns:
            T: (B, in_features, degree+1)
        """
        T = [torch.ones_like(x), x.clone()]
        for _ in range(2, self.degree + 1):
            T.append(2.0 * x * T[-1] - T[-2])
        return torch.stack(T, dim=-1)  # (B, in_features, degree+1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_features)
        Returns:
            (B, out_features)
        """
        # Normalise to [−1, 1]
        x_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        x_norm = (x / x_max).clamp(-1.0, 1.0)

        basis = self._chebyshev_basis(x_norm)  # (B, in_features, degree+1)

        # y_j = einsum("bik, oik -> bo", basis, coeffs)
        out = torch.einsum("bik, oik -> bo", basis, self.coeffs)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# KANForecaster
# ──────────────────────────────────────────────────────────────────────────────


class KANForecaster(nn.Module):
    """KAN-based forecaster (channel-independent)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        hidden: int = 64,
        e_layers: int = 2,
        degree: int = 5,
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

        layers: list[nn.Module] = []
        in_dim = seq_len
        for _ in range(e_layers):
            layers.append(_ChebyKANLayer(in_dim, hidden, degree))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        # Final KAN head
        layers.append(_ChebyKANLayer(in_dim, pred_len, degree))
        self.kan_net = nn.Sequential(*layers)

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

        x_ci = x.transpose(1, 2).reshape(B * C, T)          # (BC, T)
        out = self.kan_net(x_ci)                             # (BC, pred_len)
        out = out.reshape(B, C, self.pred_len).transpose(1, 2)  # (B, pred_len, C)

        if self._revin_flag:
            out = self.revin_layer(out, "denorm")
        return out
