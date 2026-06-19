"""SincNetForecaster: Learnable sinc bandpass filters for time-series forecasting.

Key idea (Ravanelli & Bengio, 2018 — "Speaker Recognition from Raw Waveform
with SincNet"):

Replace the first convolutional layer's arbitrary kernels with *windowed sinc*
bandpass filters.  Each filter is fully determined by two scalar parameters:

    f₁  — lower cutoff frequency (Hz, normalised to Nyquist = 0.5)
    f₂  — upper cutoff frequency

The time-domain filter of length M is:

    h[n] = 2f₂ sinc(2πf₂n) − 2f₁ sinc(2πf₁n)          ideal bandpass
    h̃[n] = h[n] · w[n]                                  Hamming-windowed

where n ∈ {−(M−1)/2, …, (M−1)/2} and sinc(x) = sin(x)/x.

The cutoff frequencies are learned via:

    f₁ₖ = |α₁ₖ| · 0.5                                    ∈ (0, 0.5)
    f₂ₖ = f₁ₖ + |α₂ₖ| · 0.5                             ∈ (f₁ₖ, 1.0)

where α₁, α₂ ∈ ℝ are unconstrained learnable scalars.
Both are clamped to (ε, 0.499) to avoid degenerate filters.

This creates a frequency-selective first stage: each filter automatically
specialises to a frequency band.  Subsequent layers are standard conv.

Architecture comparison:
  AdaptiveSpectralForecaster: learns frequency band energy in the *frequency domain*.
  SincNetForecaster:          learns bandpass filters in the *time domain* (sinc).
  MultiscaleConvForecaster:   unconstrained kernels at multiple scales.
  WaveNet:                    unconstrained gated dilated conv.

Architecture::

    Input (B, T, C)
      → RevIN norm
      → CI reshape → (B·C, 1, T)
      → SincConv1d: n_filters learnable bandpass filters → (B·C, n_filters, T)
      → BatchNorm + LeakyReLU
      → n_conv_layers × Conv1d(n_filters, n_filters, 3, padding=1) + BN + LReLU
      → global mean-pool → (B·C, n_filters)
      → Linear head → (B·C, pred_len)
      → reshape → (B, pred_len, C)
      → RevIN denorm

Args:
    seq_len:      input lookback T.
    pred_len:     forecast horizon.
    enc_in:       number of variates C.
    n_filters:    number of learnable sinc filter banks.
    kernel_size:  length M of each sinc filter (must be odd).
    n_conv_layers: number of standard conv layers after sinc stage.
    dropout:      dropout probability.
    revin:        use RevIN instance normalisation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.revin import RevIN


class _SincConv1d(nn.Module):
    """1-D sinc-filter convolution layer.

    Parameterises N bandpass sinc filters by their lower and upper cutoff
    frequencies (both learnable, constrained to be valid).
    """

    def __init__(self, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Unconstrained learnable parameters
        # α₁ ∈ ℝ → f₁ = |α₁| * 0.5     (lower cutoff, ∈ (0, 0.5))
        # α₂ ∈ ℝ → f₂ = f₁ + |α₂| * 0.5 (upper cutoff, > f₁)
        init_f1 = torch.linspace(0.02, 0.45, out_channels)
        init_f2 = torch.linspace(0.05, 0.48, out_channels)
        self.alpha1 = nn.Parameter((init_f1 / 0.5).clone())   # f1 = |α1| * 0.5
        self.alpha2 = nn.Parameter(((init_f2 - init_f1) / 0.5).clone())  # df = |α2|*0.5

        # Pre-compute Hamming window and time indices (not parameters)
        n = (kernel_size - 1) / 2
        t = torch.arange(-n, n + 1, dtype=torch.float32) / kernel_size
        hamming = 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(kernel_size) / (kernel_size - 1))
        self.register_buffer("t", t)
        self.register_buffer("hamming", hamming)

    @staticmethod
    def _sinc(x: torch.Tensor) -> torch.Tensor:
        """Normalised sinc: sin(πx)/(πx), sinc(0)=1."""
        eps = torch.finfo(x.dtype).eps
        x = torch.where(x.abs() < eps, torch.ones_like(x) * eps, x)
        return torch.sin(math.pi * x) / (math.pi * x)

    def _compute_filters(self) -> torch.Tensor:
        """Return (out_channels, 1, kernel_size) filter bank."""
        f1 = self.alpha1.abs().clamp(0.001, 0.499) * 0.5   # (N,)
        df = self.alpha2.abs().clamp(0.001, 0.499) * 0.5    # (N,)
        f2 = (f1 + df).clamp(max=0.499)                     # (N,)

        t = self.t   # (M,)
        # Ideal bandpass: 2f₂sinc(2f₂t) - 2f₁sinc(2f₁t)
        h = 2 * f2.unsqueeze(1) * self._sinc(2 * f2.unsqueeze(1) * t.unsqueeze(0)) \
          - 2 * f1.unsqueeze(1) * self._sinc(2 * f1.unsqueeze(1) * t.unsqueeze(0))
        # (N, M)

        # Hamming windowing
        h = h * self.hamming.unsqueeze(0)   # (N, M)
        return h.unsqueeze(1)               # (N, 1, M)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T) → (B, N, T)"""
        filters = self._compute_filters()
        pad = self.kernel_size // 2
        return F.conv1d(x, filters, padding=pad)


class SincNetForecaster(nn.Module):
    """SincNet bandpass-filter forecaster (channel-independent)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        n_filters: int = 32,
        kernel_size: int = 25,
        n_conv_layers: int = 2,
        dropout: float = 0.1,
        revin: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1   # force odd

        self.pred_len = pred_len
        self.enc_in = enc_in
        self._revin_flag = revin

        if revin:
            self.revin_layer = RevIN(enc_in, affine=True)

        # Sinc filter stage
        self.sinc_conv = _SincConv1d(n_filters, kernel_size)
        self.sinc_bn = nn.BatchNorm1d(n_filters)

        # Standard conv layers
        conv_layers: list[nn.Module] = []
        for _ in range(n_conv_layers):
            conv_layers += [
                nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(n_filters),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
            ]
        self.conv_layers = nn.Sequential(*conv_layers)

        self.head = nn.Linear(n_filters, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, pred_len, C)"""
        B, T, C = x.shape

        if self._revin_flag:
            x = self.revin_layer(x, "norm")

        # CI: each channel as an independent 1-D sequence
        x_ci = x.permute(0, 2, 1).reshape(B * C, T).unsqueeze(1)   # (BC, 1, T)

        # Sinc stage
        h = self.sinc_conv(x_ci)                                    # (BC, n_filters, T)
        h = F.leaky_relu(self.sinc_bn(h), 0.2)

        # Standard conv layers
        h = self.conv_layers(h)                                     # (BC, n_filters, T)

        ctx = h.mean(dim=-1)                                        # (BC, n_filters)
        pred = self.head(ctx)                                       # (BC, pred_len)
        pred = pred.reshape(B, C, self.pred_len).permute(0, 2, 1)

        if self._revin_flag:
            pred = self.revin_layer(pred, "denorm")
        return pred
