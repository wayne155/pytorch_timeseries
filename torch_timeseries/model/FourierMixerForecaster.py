"""FourierMixerForecaster: Learnable complex MLP mixing in the DFT frequency domain.

Key idea:
  Most time-series models operate on the raw time domain.  This model operates
  entirely in the DFT frequency domain:

    1. Apply the real-FFT: x ∈ ℝ^T  →  X ∈ ℂ^{T//2+1}
       (keeping only the positive-frequency half thanks to conjugate symmetry)
    2. Mix the frequency bins using independent learned linear layers for the
       real and imaginary parts:
           Re_out = W_r · Re_in  +  b_r
           Im_out = W_i · Im_in  +  b_i
    3. Apply inverse real-FFT: X_out ∈ ℂ^{T//2+1}  →  x̃ ∈ ℝ^T
    4. Linear head: ℝ^T → ℝ^{pred_len}.

  Stacking L such frequency-mixing layers with residual connections and
  layer-norm gives a purely spectral backbone.

Architecture comparison:
  FreTS:              complex-valued Fourier MLP on global frequency representation.
  FITS:               sparse interpolation on low-frequency DFT coefficients.
  HarmonicForecaster: soft selection of K frequency components + analytic extrapolation.
  FourierMixerForecaster: full-spectrum complex MLP mixing + IFFT reconstruction (CI).

Architecture::

    Input (B, T, C)
      → RevIN norm
      → CI reshape → (B·C, T)
      → L × [ rfft → Re/Im MLP → irfft → LayerNorm + residual ]
      → Linear head → (B·C, pred_len)
      → reshape → (B, pred_len, C)
      → RevIN denorm

Args:
    seq_len:    input lookback T.
    pred_len:   forecast horizon.
    enc_in:     number of variates C.
    e_layers:   number of frequency-mixing layers.
    dropout:    dropout applied after each IFFT reconstruction.
    revin:      use RevIN instance normalisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.revin import RevIN


class _FreqMixLayer(nn.Module):
    """Single frequency-domain mixing layer."""

    def __init__(self, seq_len: int, dropout: float = 0.1):
        super().__init__()
        n_freq = seq_len // 2 + 1
        # Separate linear mixers for real and imaginary parts
        self.mix_r = nn.Linear(n_freq, n_freq, bias=True)
        self.mix_i = nn.Linear(n_freq, n_freq, bias=True)
        self.norm = nn.LayerNorm(seq_len)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) real signal → (B, T) after freq mixing."""
        # DFT
        X = torch.fft.rfft(x, dim=-1)          # (B, T//2+1) complex

        # Mix real and imaginary parts independently
        r_out = self.mix_r(X.real)
        i_out = self.mix_i(X.imag)
        X_mixed = torch.complex(r_out, i_out)

        # IDFT
        x_rec = torch.fft.irfft(X_mixed, n=x.shape[-1], dim=-1)   # (B, T)
        x_rec = self.drop(x_rec)

        # Residual + norm
        return self.norm(x + x_rec)


class FourierMixerForecaster(nn.Module):
    """Stacked DFT frequency-domain MLP mixer (channel-independent)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        e_layers: int = 3,
        dropout: float = 0.1,
        revin: bool = True,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in
        self._revin_flag = revin

        if revin:
            self.revin_layer = RevIN(enc_in, affine=True)

        self.layers = nn.ModuleList(
            [_FreqMixLayer(seq_len, dropout) for _ in range(e_layers)]
        )
        self.head = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, pred_len, C)"""
        B, T, C = x.shape

        if self._revin_flag:
            x = self.revin_layer(x, "norm")

        # CI: each channel independently
        x_ci = x.permute(0, 2, 1).reshape(B * C, T)   # (BC, T)

        for layer in self.layers:
            x_ci = layer(x_ci)

        pred = self.head(x_ci)                         # (BC, pred_len)
        pred = pred.reshape(B, C, self.pred_len).permute(0, 2, 1)   # (B, pred_len, C)

        if self._revin_flag:
            pred = self.revin_layer(pred, "denorm")
        return pred
