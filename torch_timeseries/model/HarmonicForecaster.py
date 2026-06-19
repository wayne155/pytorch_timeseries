"""HarmonicForecaster: Differentiable spectral decomposition for time-series forecasting.

Key idea:
  Every channel is modelled as a superposition of K sinusoidal components whose
  frequencies, amplitudes, and phases are *extracted* from the input spectrum via
  a soft attention mechanism, then analytically extrapolated to the horizon:

      x̂[t] = Σ_{k=1}^{K}  A_k · cos(2π f_k t / T  +  φ_k)  +  trend(t)

  where trend(t) is a channel-wise linear regression on the lookback window.

  Instead of learning fixed frequencies, we compute the DFT of each input channel
  and use K learnable soft-selectors (softmax-normalised weights over frequency bins)
  to pick a weighted mixture of DFT components.  All operations are differentiable.

Architecture:
  1. RevIN normalise.
  2. Per-channel rfft  →  magnitude + phase spectra.
  3. K learnable frequency queries (K × n_freq, softmax-normalised) select K
     effective (f_k, A_k, φ_k) triples from the spectrum.
  4. Each triple is extrapolated over the future time axis analytically.
  5. A linear trend estimated via least-squares from the normalised lookback is added.
  6. (Optional) a lightweight MLP residual blends the K harmonic outputs.

Args:
    seq_len:       lookback length T.
    pred_len:      forecast horizon.
    enc_in:        number of variates.
    n_harmonics:   number of learned frequency components K.
    use_mlp:       if True, add a small MLP blending head on top of harmonic sum.
    d_mlp:         hidden size of the optional MLP blending head.
    dropout:       dropout applied inside MLP head.
    revin:         use RevIN normalisation.
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
# HarmonicForecaster
# ──────────────────────────────────────────────────────────────────────────────


class HarmonicForecaster(nn.Module):
    """Differentiable spectral decomposition forecaster (channel-independent)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        n_harmonics: int = 16,
        use_mlp: bool = True,
        d_mlp: int = 64,
        dropout: float = 0.1,
        revin: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.n_harmonics = n_harmonics
        self._revin_flag = revin

        n_freq = seq_len // 2 + 1
        self.n_freq = n_freq

        if revin:
            self.revin_layer = _RevIN(enc_in)

        # K learnable soft frequency selectors: (K, n_freq), normalised with softmax
        self.freq_queries = nn.Parameter(torch.randn(n_harmonics, n_freq))

        # Frequency index buffer: shape (n_freq,)
        freq_idx = torch.arange(n_freq, dtype=torch.float32)
        self.register_buffer("freq_idx", freq_idx)

        # Future time axis: positions T, T+1, ..., T+pred_len-1
        t_future = torch.arange(seq_len, seq_len + pred_len, dtype=torch.float32)
        self.register_buffer("t_future", t_future)

        # Past time axis for trend estimation
        t_past = torch.arange(seq_len, dtype=torch.float32)
        self.register_buffer("t_past", t_past)

        # Optional MLP blending head: takes harmonic sum and projects to pred_len
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(pred_len, d_mlp),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_mlp, pred_len),
            )
        else:
            self.mlp = None

    def _linear_trend(self, x_ci: torch.Tensor) -> torch.Tensor:
        """Least-squares linear trend per channel, extrapolated to future.

        Args:
            x_ci: (B*C, T) normalised input.
        Returns:
            trend: (B*C, pred_len)
        """
        T = x_ci.shape[-1]
        t = self.t_past  # (T,)
        t_mean = t.mean()
        x_mean = x_ci.mean(-1, keepdim=True)            # (B*C, 1)
        denom = ((t - t_mean) ** 2).sum()
        slope = ((x_ci - x_mean) * (t - t_mean)).sum(-1) / denom.clamp(min=1e-6)  # (B*C,)
        intercept = x_mean.squeeze(-1) - slope * t_mean  # (B*C,)
        trend = (
            slope.unsqueeze(-1) * self.t_future.unsqueeze(0)
            + intercept.unsqueeze(-1)
        )  # (B*C, pred_len)
        return trend

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

        x_ci = x.transpose(1, 2).reshape(B * C, T)      # (B*C, T)

        # ── 1. FFT ──────────────────────────────────────────────────────────
        coeffs = torch.fft.rfft(x_ci, norm="ortho")      # (B*C, n_freq) complex
        magnitude = coeffs.abs()                          # (B*C, n_freq)
        phase = coeffs.angle()                            # (B*C, n_freq)

        # ── 2. Soft frequency selection ──────────────────────────────────────
        w = F.softmax(self.freq_queries, dim=-1)          # (K, n_freq)

        # Effective amplitude, phase and frequency per component:
        #   A_k   = Σ_f w_k[f] * magnitude[f]   → (B*C, K)
        #   φ_k   = Σ_f w_k[f] * phase[f]       → (B*C, K)
        #   f_k   = Σ_f w_k[f] * f               → (K,)
        A = magnitude @ w.T                               # (B*C, K)
        phi = phase @ w.T                                 # (B*C, K)
        f_eff = w @ self.freq_idx                         # (K,)

        # ── 3. Analytic extrapolation ────────────────────────────────────────
        # angle[k, t] = 2π * f_k * t_future[t] / T
        angles = (
            2.0 * math.pi / T
            * f_eff.unsqueeze(-1)            # (K, 1)
            * self.t_future.unsqueeze(0)     # (1, pred_len)
        )  # (K, pred_len)

        # harmonics[b*c, k, t] = A_k * cos(angles[k, t] + phi_k)
        harmonic_vals = A.unsqueeze(-1) * torch.cos(
            angles.unsqueeze(0) + phi.unsqueeze(-1)
        )  # (B*C, K, pred_len)

        harmonic_sum = harmonic_vals.sum(dim=1)           # (B*C, pred_len)

        # ── 4. Linear trend ─────────────────────────────────────────────────
        trend = self._linear_trend(x_ci)                  # (B*C, pred_len)

        out = harmonic_sum + trend                        # (B*C, pred_len)

        # ── 5. Optional MLP residual ─────────────────────────────────────────
        if self.mlp is not None:
            out = out + self.mlp(out)

        out = out.reshape(B, C, self.pred_len).transpose(1, 2)  # (B, pred_len, C)

        if self._revin_flag:
            out = self.revin_layer(out, "denorm")
        return out
