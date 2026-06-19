"""FiLM: Frequency improved Legendre Memory for time-series forecasting.

Reference: Zhou et al., "FiLM: Frequency improved Legendre Memory for
Long-term Time Series Forecasting", NeurIPS 2022.

Key ideas:
  Standard MLPs applied directly to T input steps require O(T) parameters
  per layer and are sensitive to noise.  FiLM addresses this with two steps:

    1. **Legendre Memory Projection**: project the T-length input window onto
       a Legendre polynomial basis of order N (N << T).  The Legendre
       polynomials are orthogonal over [-1, 1] and give a compact, low-noise
       representation.  Implemented as a fixed (non-learnable) projection
       matrix derived from the Legendre recurrence, inspired by HiPPO.

    2. **Frequency Low-pass Filter**: apply FFT to the N Legendre coefficients,
       zero out the top-K high-frequency modes, and reconstruct with IFFT.
       This removes residual high-frequency noise in the polynomial space.

  Final decoder: MLP from the filtered N-dimensional representation to
  pred_len × enc_in, applied channel-independently.

  Optional RevIN normalisation on input/output.

Args:
    seq_len:   input lookback window.
    pred_len:  forecast horizon.
    enc_in:    number of variates.
    d_order:   number of Legendre polynomial basis functions (order N).
    n_lowpass: number of low-frequency Fourier modes to *keep* (rest zeroed).
    d_ff:      MLP hidden size.
    dropout:   dropout rate.
    revin:     use RevIN instance normalisation.
"""
from __future__ import annotations

import math

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


def _legendre_matrix(seq_len: int, order: int) -> torch.Tensor:
    """Build the Legendre projection matrix W ∈ R^{order × seq_len}.

    W[n, t] = L_n(2t/(T-1) - 1) * (2n+1) / T  where L_n is the n-th
    Legendre polynomial.  The (2n+1)/T factor is the quadrature weight so
    that the projection is the polynomial coefficient, not just the sample.
    """
    T = seq_len
    # Evaluate each Legendre polynomial at T evenly-spaced points in [-1, 1]
    xs = torch.linspace(-1, 1, T)  # (T,)
    W = torch.zeros(order, T)
    # Recurrence: P_0 = 1, P_1 = x, P_{n+1} = ((2n+1)xP_n - nP_{n-1})/(n+1)
    if order >= 1:
        W[0] = 1.0
    if order >= 2:
        W[1] = xs
    for n in range(2, order):
        W[n] = ((2 * n - 1) * xs * W[n - 1] - (n - 1) * W[n - 2]) / n
    # Scale by (2n+1)/T (quadrature weight)
    for n in range(order):
        W[n] = W[n] * (2 * n + 1) / T
    return W  # (order, seq_len)


class FiLM(nn.Module):
    """Frequency improved Legendre Memory (channel-independent)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_order: int = 32,
        n_lowpass: int = 2,
        d_ff: int = 256,
        dropout: float = 0.05,
        revin: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_order = d_order
        self.n_lowpass = n_lowpass
        self._revin_flag = revin
        if revin:
            self.revin_layer = _RevIN(enc_in)

        # Legendre projection matrix (fixed, not learned)
        W = _legendre_matrix(seq_len, d_order)  # (d_order, seq_len)
        self.register_buffer("W", W)

        # MLP decoder: d_order → pred_len, applied per variate
        self.decoder = nn.Sequential(
            nn.Linear(d_order, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, pred_len),
        )

    def _low_pass_filter(self, coeff: torch.Tensor) -> torch.Tensor:
        """Zero out high-frequency modes beyond n_lowpass.

        Args:
            coeff: (B, C, d_order)
        Returns:
            (B, C, d_order) with high-frequency modes zeroed
        """
        X = torch.fft.rfft(coeff, dim=-1, norm="ortho")    # (B, C, d_order//2+1)
        # keep only the first n_lowpass modes
        if self.n_lowpass < X.shape[-1]:
            X[..., self.n_lowpass:] = 0
        return torch.fft.irfft(X, n=self.d_order, dim=-1, norm="ortho")  # (B, C, d_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)
        Returns:
            (B, pred_len, C)
        """
        if self._revin_flag:
            x = self.revin_layer(x, "norm")

        B, T, C = x.shape
        # x: (B, T, C) → transpose → (B, C, T)
        x_t = x.transpose(1, 2)  # (B, C, T)

        # Legendre projection: (B, C, T) @ W^T → (B, C, d_order)
        coeff = x_t @ self.W.T  # (B, C, d_order)

        # Frequency low-pass filter
        coeff = self._low_pass_filter(coeff)  # (B, C, d_order)

        # MLP decode: (B, C, d_order) → (B, C, pred_len)
        out = self.decoder(coeff)

        # (B, C, pred_len) → (B, pred_len, C)
        out = out.transpose(1, 2)

        if self._revin_flag:
            out = self.revin_layer(out, "denorm")
        return out
