"""RandomFourierForecaster: Random Fourier Feature kernel approximation forecaster.

Key idea (Rahimi & Recht, NeurIPS 2007):
  Any shift-invariant kernel k(x, y) = k(x−y) can be approximated by an explicit
  randomised feature map φ : ℝ^d → ℝ^D such that:

      k(x, y) ≈ φ(x)^T φ(y)

  For the Gaussian (RBF) kernel k(x,y) = exp(−||x−y||²/(2σ²)) the map is:

      W ~ N(0, I/σ²)   shape (D, T)
      b ~ Uniform(0, 2π)  shape (D,)
      φ(x) = √(2/D) · cos(x·W^T + b)   shape (D,)

  The entire feature map is fixed after initialisation (no gradients flow through W or b).
  Only a linear readout from the D-dimensional RFF space to pred_len is trained, making
  this a kernel-regression baseline implemented as a neural module.

Architecture comparison:
  TSReservoir:            fixed random recurrent (echo state) dynamics → linear readout.
  RandomFourierForecaster: fixed random spectral (Fourier) lifting → linear readout.
  LinearAttentionForecaster: learned ELU+1 kernel approximation in attention.

Architecture::

    Input (B, T, C)
      → RevIN norm
      → CI reshape → (B·C, T)
      → RFF φ(x) = √(2/D) cos(x·W^T + b)  → (B·C, D)   [W, b fixed]
      → Linear(D, pred_len)                → (B·C, pred_len)
      → reshape → (B, pred_len, C)
      → RevIN denorm

Args:
    seq_len:    input lookback T.
    pred_len:   forecast horizon.
    enc_in:     number of variates C.
    d_rff:      number of random Fourier features D (projection dimension).
    sigma:      length-scale of the RBF kernel; larger σ → smoother kernel.
    revin:      use RevIN instance normalisation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from torch_timeseries.nn.revin import RevIN


class RandomFourierForecaster(nn.Module):
    """Fixed RFF kernel approximation with a single linear readout (channel-independent)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_rff: int = 256,
        sigma: float = 1.0,
        revin: bool = True,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in
        self._revin_flag = revin

        if revin:
            self.revin_layer = RevIN(enc_in, affine=True)

        # Fixed random projection — not parameters, not updated by optimizer
        W = torch.randn(d_rff, seq_len) / sigma          # (D, T)
        b = torch.rand(d_rff) * (2 * math.pi)            # (D,)
        self.register_buffer("W", W)
        self.register_buffer("b", b)

        # Only trained component
        self.readout = nn.Linear(d_rff, pred_len)

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

        # CI: treat each variate independently
        x_ci = x.permute(0, 2, 1).reshape(B * C, T)      # (BC, T)

        # Random Fourier Feature map
        proj = x_ci @ self.W.T + self.b                  # (BC, D)
        d_rff = self.W.shape[0]
        phi = math.sqrt(2.0 / d_rff) * torch.cos(proj)  # (BC, D)

        # Linear readout
        pred = self.readout(phi)                          # (BC, pred_len)
        pred = pred.reshape(B, C, self.pred_len).permute(0, 2, 1)  # (B, pred_len, C)

        if self._revin_flag:
            pred = self.revin_layer(pred, "denorm")
        return pred
