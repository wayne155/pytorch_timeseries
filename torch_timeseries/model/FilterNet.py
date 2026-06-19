"""FilterNet: Learnable Frequency-domain Filter Bank for time-series forecasting.

Reference: Han et al., "FilterNet: Harnessing Frequency Filters for Time Series
Forecasting", NeurIPS 2024.

Key ideas:
  Frequency-domain methods like FEDformer and FiLM show that time series often
  have low-dimensional frequency structure.  FilterNet builds on this by:

    1. **Learnable Band-pass Filters**: instead of a fixed top-K FFT truncation
       or random frequency selection, FilterNet learns *filter weights* w_k for
       each frequency component k.  The filter is applied multiplicatively in
       the frequency domain:
           X_filtered[k] = X[k] * w_k
       where w_k ∈ [0, 1] (soft selection) or ∈ R (general linear filter).

    2. **Multi-filter Bank**: multiple parallel filters capture different
       frequency bands (low-pass, band-pass, high-pass).  Their outputs are
       mixed by a learnable gating mechanism.

    3. **Channel-independent processing**: each variate is filtered
       independently, keeping the model lightweight.

    4. **Residual connection**: the original time-domain signal is added back
       after filtering, ensuring the model can at minimum learn identity.

  Pipeline:
    x → RevIN → FFT → per-filter learnable weighting → IFFT → gate-mix
    → linear decoder → pred_len → RevIN-denorm

  The filter weights are real-valued and applied to the complex FFT spectrum
  (multiplied elementwise to both real and imaginary parts).  Optionally they
  can be complex (full linear transform per frequency bin).

Args:
    seq_len:     input lookback window.
    pred_len:    forecast horizon.
    enc_in:      number of variates.
    num_filters: number of filters in the bank.
    revin:       use RevIN instance normalisation.
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


class _FrequencyFilter(nn.Module):
    """Single learnable frequency filter.

    For each of the F = seq_len//2+1 frequency bins, a learnable real-valued
    weight controls how much of that frequency to keep.  Sigmoid ensures the
    weight is in (0, 1) — a soft band selector.
    """

    def __init__(self, n_freq: int):
        super().__init__()
        # Initialise near 1 so the filter starts as near-identity
        self.weights = nn.Parameter(torch.ones(n_freq) * 2.0)  # sigmoid(2) ≈ 0.88

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: complex tensor (B_C, F) — rfft output
        Returns:
            (B_C, F) complex — filtered spectrum
        """
        w = torch.sigmoid(self.weights)    # (F,) in (0,1)
        return X * w                       # broadcast over B_C


class FilterNet(nn.Module):
    """FilterNet: learnable frequency-domain filter bank (channel-independent).

    Multiple parallel frequency filters → linear combination → time-domain
    → linear decoder → pred_len.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        num_filters: int = 8,
        revin: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.num_filters = num_filters
        self._revin_flag = revin
        if revin:
            self.revin_layer = _RevIN(enc_in)

        # Number of frequency bins for rfft
        n_freq = seq_len // 2 + 1
        self.n_freq = n_freq

        # Filter bank: num_filters independent frequency filters
        self.filters = nn.ModuleList(
            [_FrequencyFilter(n_freq) for _ in range(num_filters)]
        )

        # Gating: learned linear combination of filter outputs (time domain)
        # Input: num_filters filtered signals (B_C, seq_len) → output (B_C, seq_len)
        self.gate = nn.Linear(num_filters, 1)
        nn.init.constant_(self.gate.weight, 1.0 / num_filters)
        nn.init.zeros_(self.gate.bias)

        # Residual mix: blend filtered + original
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Decoder: seq_len → pred_len, applied per-variate
        self.decoder = nn.Linear(seq_len, pred_len)

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

        # Channel-independent: (B, T, C) → (B, C, T) → (B*C, T)
        x_ci = x.transpose(1, 2).reshape(B * C, T)

        # FFT: (B*C, T) → (B*C, n_freq) complex
        X = torch.fft.rfft(x_ci, dim=-1, norm="ortho")

        # Apply each filter and IFFT back to time domain
        filtered = []
        for filt in self.filters:
            X_f = filt(X)                                              # (B*C, n_freq) complex
            x_f = torch.fft.irfft(X_f, n=T, dim=-1, norm="ortho")     # (B*C, T) real
            filtered.append(x_f)

        # Stack filter outputs: (B*C, num_filters, T)
        filtered_stack = torch.stack(filtered, dim=1)

        # Gated mix: (B*C, T, num_filters) → (B*C, T, 1) → (B*C, T)
        mixed = self.gate(filtered_stack.transpose(1, 2)).squeeze(-1)  # (B*C, T)

        # Residual blend
        alpha = torch.sigmoid(self.alpha)
        h = alpha * mixed + (1 - alpha) * x_ci

        # Decode: (B*C, T) → (B*C, pred_len)
        out = self.decoder(h)

        # Reshape: (B*C, pred_len) → (B, C, pred_len) → (B, pred_len, C)
        out = out.reshape(B, C, self.pred_len).transpose(1, 2)

        if self._revin_flag:
            out = self.revin_layer(out, "denorm")
        return out
