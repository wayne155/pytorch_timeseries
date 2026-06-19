"""AdaptiveSpectralForecaster: Learnable soft bandpass spectral filtering.

Key idea:
  Instead of mixing all DFT coefficients through an MLP (FourierMixerForecaster),
  this model learns S explicit *bandpass filters* parameterised by their centre
  frequency μ_s and bandwidth σ_s:

      mask_s(f) = sigmoid( −(f − μ_s)² / (2 σ_s²) )     s = 1, …, S

  Each filter selects a soft band of the DFT spectrum.  The energy (L2 norm of
  the selected complex coefficients) within each band is computed, giving an
  S-dimensional feature vector that captures dominant spectral components.
  A linear head maps these S features to the forecast horizon.

  Distinct inductive bias: no dense mixing, no IIR/FIR convolution — just
  learned frequency-band energy extraction followed by linear regression.

Architecture comparison:
  FourierMixerForecaster:       MLP mixes all DFT bins.
  FITS:                         sparse low-frequency DFT interpolation.
  HarmonicForecaster:           soft frequency *selection* for extrapolation.
  AdaptiveSpectralForecaster:   learnable bandpass *energy* extraction → linear head.

Architecture::

    Input (B, T, C)
      → RevIN norm
      → CI reshape → (B·C, T)
      → rfft → X ∈ ℂ^{T//2+1}
      → S bandpass masks mask_s(f) = sigmoid(−(f−μ)²/(2σ²))
      → band energies e_s = ||mask_s ⊙ X||_2    → (B·C, S)
      → Linear(S, pred_len) → (B·C, pred_len)
      → reshape → (B, pred_len, C)
      → RevIN denorm

Args:
    seq_len:    input lookback T.
    pred_len:   forecast horizon.
    enc_in:     number of variates C.
    n_filters:  number of learnable bandpass filters S.
    revin:      use RevIN instance normalisation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from torch_timeseries.nn.revin import RevIN


class AdaptiveSpectralForecaster(nn.Module):
    """Learnable soft bandpass spectral filtering (channel-independent)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        n_filters: int = 16,
        revin: bool = True,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in
        self._revin_flag = revin
        self.n_freq = seq_len // 2 + 1

        if revin:
            self.revin_layer = RevIN(enc_in, affine=True)

        # Learnable bandpass centre (μ) and log-bandwidth (log σ) per filter
        # Initialise μ uniformly over [0, n_freq), σ = 1 (log σ = 0)
        init_mu = torch.linspace(0, self.n_freq - 1, n_filters)
        self.mu = nn.Parameter(init_mu)
        self.log_sigma = nn.Parameter(torch.zeros(n_filters))

        # Frequency index buffer (not trained)
        self.register_buffer("freq_idx", torch.arange(self.n_freq, dtype=torch.float32))

        # Linear readout from S band energies to pred_len
        self.head = nn.Linear(n_filters, pred_len)

    def _bandpass_masks(self) -> torch.Tensor:
        """Compute S bandpass masks over n_freq bins.

        Returns:
            masks: (S, n_freq) in (0, 1)
        """
        sigma = self.log_sigma.exp().clamp(min=1e-3)               # (S,)
        diff = self.freq_idx.unsqueeze(0) - self.mu.unsqueeze(1)   # (S, n_freq)
        exponent = -(diff ** 2) / (2 * sigma.unsqueeze(1) ** 2)
        return torch.sigmoid(exponent)                             # (S, n_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, pred_len, C)"""
        B, T, C = x.shape

        if self._revin_flag:
            x = self.revin_layer(x, "norm")

        # CI
        x_ci = x.permute(0, 2, 1).reshape(B * C, T)              # (BC, T)

        # DFT
        X = torch.fft.rfft(x_ci, dim=-1)                          # (BC, n_freq) complex

        # Bandpass energy: ||mask_s ⊙ X||_2  for each filter s
        masks = self._bandpass_masks()                             # (S, n_freq)
        # X_abs: (BC, n_freq), masks: (S, n_freq) → (BC, S) via einsum
        X_abs = X.abs()                                            # (BC, n_freq)
        band_energy = torch.einsum("bf,sf->bs", X_abs, masks)     # (BC, S)

        # Head
        pred = self.head(band_energy)                              # (BC, pred_len)
        pred = pred.reshape(B, C, self.pred_len).permute(0, 2, 1) # (B, pred_len, C)

        if self._revin_flag:
            pred = self.revin_layer(pred, "denorm")
        return pred
