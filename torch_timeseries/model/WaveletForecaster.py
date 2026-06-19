"""WaveletForecaster: Multi-resolution wavelet decomposition for forecasting.

Key idea:
  Time-series typically contain patterns at multiple temporal resolutions (trends,
  seasonal, high-frequency noise).  A wavelet decomposition explicitly separates
  these by iteratively applying half-band filters:

    Haar DWT level j:
      A_{j+1}[n] = (A_j[2n]   + A_j[2n+1]) / √2    (approximation)
      D_{j+1}[n] = (A_j[2n]   - A_j[2n+1]) / √2    (detail / high-freq residual)

  After J levels we obtain bands [D_1, D_2, ..., D_J, A_J] of lengths
  [T/2, T/4, ..., T/2^J, T/2^J].  Each band captures signal energy at a
  different temporal scale.

  Instead of forecasting in the original domain, WaveletForecaster:
    1. Decomposes the input into J+1 bands via J-level Haar DWT.
    2. Applies a separate learnable linear head to each band: (T/2^k → pred_len).
    3. Sums the J+1 band-level predictions.

  Because each head operates on a coarse/fine version of the signal, the model
  implicitly learns which temporal scales matter for the horizon.

Architecture comparison:
  FilterNet:  learns frequency-domain weights on the rfft spectrum (all freqs at once).
  HarmonicForecaster: extracts K sinusoidal components from spectrum.
  WaveletForecaster: explicit multi-resolution DWT — orthogonal, invertible, exact.

Args:
    seq_len:    lookback length T.
    pred_len:   forecast horizon.
    enc_in:     number of variates.
    n_levels:   number of DWT levels J (default 3).
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
# Single-level Haar DWT
# ──────────────────────────────────────────────────────────────────────────────


class _HaarDWT1d(nn.Module):
    """Single-level Haar discrete wavelet transform (fixed, not trained)."""

    def __init__(self):
        super().__init__()
        s = 1.0 / math.sqrt(2)
        h = torch.tensor([[[s, s]]])   # (1, 1, 2) low-pass
        g = torch.tensor([[[s, -s]]])  # (1, 1, 2) high-pass
        self.register_buffer("h", h)
        self.register_buffer("g", g)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T)
        Returns:
            approx: (B, T//2)   — low-frequency
            detail: (B, T//2)   — high-frequency
        """
        if x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, 1))
        xu = x.unsqueeze(1)                              # (B, 1, T)
        approx = F.conv1d(xu, self.h, stride=2).squeeze(1)
        detail = F.conv1d(xu, self.g, stride=2).squeeze(1)
        return approx, detail


# ──────────────────────────────────────────────────────────────────────────────
# WaveletForecaster
# ──────────────────────────────────────────────────────────────────────────────


class WaveletForecaster(nn.Module):
    """Multi-resolution wavelet decomposition forecaster (channel-independent)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        n_levels: int = 3,
        revin: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.n_levels = n_levels
        self._revin_flag = revin

        if revin:
            self.revin_layer = _RevIN(enc_in)

        self.dwt = _HaarDWT1d()

        # Compute the band lengths after J levels
        band_lengths = []
        length = seq_len
        for _ in range(n_levels):
            length = (length + 1) // 2       # ceil (DWT pads odd-length inputs)
            band_lengths.append(length)      # detail at this level has `length` samples

        # band_lengths[-1] also equals the approx length at the final level
        # bands: [detail_1, detail_2, ..., detail_J, approx_J]
        all_lengths = band_lengths + [band_lengths[-1]]
        self.band_heads = nn.ModuleList(
            [nn.Linear(ln, pred_len) for ln in all_lengths]
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

        x_ci = x.transpose(1, 2).reshape(B * C, T)          # (BC, T)

        approx = x_ci
        details: list[torch.Tensor] = []
        for _ in range(self.n_levels):
            approx, detail = self.dwt(approx)
            details.append(detail)

        # Forecast each band separately
        bands = details + [approx]                           # J details + final approx
        out = sum(head(band) for head, band in zip(self.band_heads, bands))
        # out: (BC, pred_len)

        out = out.reshape(B, C, self.pred_len).transpose(1, 2)  # (B, pred_len, C)

        if self._revin_flag:
            out = self.revin_layer(out, "denorm")
        return out
