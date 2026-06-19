"""NeuralBasisForecaster: Learnable basis-function decomposition for forecasting.

Key idea:

Project the input time series onto K learned encoder bases, nonlinearly
transform the K-dimensional coefficient vector, then reconstruct via K learned
decoder bases:

    Φ_enc ∈ ℝ^{K×T}   — encoder basis (learned, L2-normalised at runtime)
    Φ_dec ∈ ℝ^{K×H}   — decoder basis (learned)

    c    = x · Φ_enc^T     ∈ ℝ^K   (analysis: project onto encoder bases)
    ĉ    = MLP(c)           ∈ ℝ^K   (nonlinear transform in basis space)
    ŷ    = ĉ · Φ_dec        ∈ ℝ^H   (synthesis: project back to forecast domain)

The basis functions Φ_enc and Φ_dec are optimised end-to-end via gradient
descent — their shapes adapt to the training data.

This places NeuralBasisForecaster in a different niche from other basis models:

    FiLM:                   fixed Fourier basis (sinusoids); learnable only downstream.
    RandomFourierForecaster: fixed random cosine basis (frozen at init).
    NBEATS:                 predefined polynomial trend + Fourier seasonality bases.
    NeuralBasisForecaster:  fully LEARNED encoder and decoder bases.

The K-dimensional bottleneck provides explicit dimensionality reduction, which
acts as a built-in regulariser: the model must represent the input compactly
before forecasting.

Architecture::

    Input (B, T, C)
      → RevIN norm
      → CI reshape → (B·C, T)
      → normalised encoder basis Φ_enc projection: (B·C, K)   [inner products]
      → coefficient LayerNorm
      → MLP [K → d_hidden → K] in basis space
      → decoder basis Φ_dec synthesis: (B·C, pred_len)
      → reshape → (B, pred_len, C)
      → RevIN denorm

Args:
    seq_len:   input lookback T.
    pred_len:  forecast horizon (decoder basis output length H).
    enc_in:    number of variates C.
    n_basis:   number of basis functions K.
    d_hidden:  hidden width of the coefficient-space MLP.
    dropout:   dropout probability.
    revin:     use RevIN instance normalisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.revin import RevIN


class NeuralBasisForecaster(nn.Module):
    """Learnable basis-function decomposition forecaster (channel-independent)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        n_basis: int = 32,
        d_hidden: int = 64,
        dropout: float = 0.1,
        revin: bool = True,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in
        self._revin_flag = revin

        if revin:
            self.revin_layer = RevIN(enc_in, affine=True)

        # Basis parameters
        # Encoder: (K, T) — project time series onto K patterns
        self.encoder_basis = nn.Parameter(torch.randn(n_basis, seq_len) * 0.02)
        # Decoder: (K, pred_len) — synthesise forecast from K coefficients
        self.decoder_basis = nn.Parameter(torch.randn(n_basis, pred_len) * 0.02)

        # Coefficient-space transform
        self.coeff_norm = nn.LayerNorm(n_basis)
        self.coeff_mlp = nn.Sequential(
            nn.Linear(n_basis, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_basis),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, pred_len, C)"""
        B, T, C = x.shape

        if self._revin_flag:
            x = self.revin_layer(x, "norm")

        # CI: each channel independently
        x_ci = x.permute(0, 2, 1).reshape(B * C, T)   # (BC, T)

        # Analysis: project onto normalised encoder basis
        # L2-normalise basis functions so scale of coefficients is comparable
        phi_enc = F.normalize(self.encoder_basis, dim=-1)   # (K, T)
        c = x_ci @ phi_enc.T                               # (BC, K)

        # Transform in basis space
        c = self.coeff_norm(c)
        c = c + self.coeff_mlp(c)                           # residual in coeff space

        # Synthesis: decode via decoder basis
        pred = c @ self.decoder_basis                       # (BC, pred_len)
        pred = pred.reshape(B, C, self.pred_len).permute(0, 2, 1)

        if self._revin_flag:
            pred = self.revin_layer(pred, "denorm")
        return pred
