"""S4Forecaster: Diagonal Structured State Space (S4D) forecaster.

Key idea (Gu et al., 2022 — "On the Parameterization and Initialization of
Diagonal State Space Models"):

A structured state space model (SSM) maps an input sequence x ∈ ℝ^T to an
output sequence y ∈ ℝ^T via a latent state h ∈ ℝ^N:

    ḣ(t) = A h(t) + B u(t)
    y(t) = C h(t) + D u(t)

For diagonal A = diag(Λ) and step size Δ (ZOH discretisation):

    Ā = exp(Δ Λ)        (element-wise, complex diagonal)
    K[t] = Re(Σ_n C_n B_n Ā_n^t)   for t = 0, 1, …, T−1

The kernel K is a sum of exponentially damped sinusoids (resonance bank).
The convolution  y = x * K + D x  is computed via FFT in O(T log T).

This implementation uses the S4D-Lin parameterisation with learnable decay
and phase for each of N state dimensions:

    Λ̄_n = exp(−softplus(ν_n) + i·π·tanh(θ_n))   |Λ̄_n| < 1 always

    K[t] = Σ_n (C_n B_n) · exp(−softplus(ν_n)·t) · cos(π·tanh(θ_n)·t)

Architecture comparison:
  MambaForecaster:  selective SSM (S6) — input-dependent A, B, C (no conv).
  LRUForecaster:    diagonal linear RNN — sequential scan, no FFT.
  S4Forecaster:     diagonal S4D — analytic kernel + FFT conv, O(T log T).

Architecture::

    Input (B, T, C)
      → RevIN norm
      → CI reshape → (B·C, T)
      → Linear embed → (B·C, T, d_model)
      → n_layers × [ S4D conv + skip + GELU MLP → LayerNorm + residual ]
      → global mean-pool → (B·C, d_model)
      → Linear head → (B·C, pred_len)
      → reshape → (B, pred_len, C)
      → RevIN denorm

Args:
    seq_len:   input lookback T.
    pred_len:  forecast horizon.
    enc_in:    number of variates C.
    d_model:   number of parallel SSMs (hidden width).
    d_state:   number of complex states per SSM (resonance bank size).
    n_layers:  number of stacked S4D + MLP blocks.
    mlp_mult:  pointwise MLP expansion factor.
    dropout:   dropout probability.
    revin:     use RevIN instance normalisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.revin import RevIN


def _fft_conv(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Linear (causal) convolution via FFT.

    Args:
        x: (B, H, T) input sequences.
        k: (H, T)    convolution kernels, one per H.

    Returns:
        (B, H, T) convolved output (first T samples).
    """
    T = x.shape[-1]
    n_fft = 2 * T
    X = torch.fft.rfft(x, n=n_fft, dim=-1)           # (B, H, n_fft//2+1)
    K = torch.fft.rfft(k, n=n_fft, dim=-1)           # (H, n_fft//2+1)
    Y = X * K.unsqueeze(0)                            # broadcast over batch
    return torch.fft.irfft(Y, n=n_fft, dim=-1)[..., :T]


class _S4DKernel(nn.Module):
    """Batched diagonal S4D kernels: d_model independent SSMs each of size d_state.

    Returns a (d_model, T) real kernel matrix computed analytically from the
    eigenvalue parameterisation.
    """

    def __init__(self, d_model: int, d_state: int, seq_len: int) -> None:
        super().__init__()
        # Stable eigenvalue parameterisation
        # log_decay → softplus(log_decay) = decay rate > 0 → |Λ| = e^{-decay} < 1
        self.log_decay = nn.Parameter(torch.zeros(d_model, d_state))
        # phase → π·tanh(phase) ∈ (−π, π)
        phase_init = torch.linspace(-1.0, 1.0, d_state).unsqueeze(0).expand(d_model, -1)
        self.phase = nn.Parameter(phase_init.clone())
        # Input/output mixing coefficients (real)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        # Time index buffer (never a gradient)
        self.register_buffer("t_idx", torch.arange(seq_len, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        """Compute and return the (d_model, T) real kernel."""
        decay = F.softplus(self.log_decay)                 # (H, N) > 0
        arg = torch.pi * torch.tanh(self.phase)            # (H, N) ∈ (−π, π)
        t = self.t_idx                                     # (T,)

        # Λ^t: (H, N, T)  — exp(−decay·t) · cos(arg·t)
        exp_decay = torch.exp(-decay.unsqueeze(-1) * t)    # (H, N, T)
        cos_arg = torch.cos(arg.unsqueeze(-1) * t)         # (H, N, T)
        lam_t = exp_decay * cos_arg                        # (H, N, T)

        # K[h, t] = Σ_n  C[h,n] B[h,n]  Λ[h,n]^t
        cb = (self.C * self.B).unsqueeze(-1)               # (H, N, 1)
        return (cb * lam_t).sum(dim=1)                     # (H, T)


class _S4DBlock(nn.Module):
    """One S4D layer: SSM convolution + skip + GELU MLP + LayerNorm + residual."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        seq_len: int,
        mlp_mult: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.kernel = _S4DKernel(d_model, d_state, seq_len)
        self.D = nn.Parameter(torch.ones(d_model))     # per-channel skip
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        # Rearrange for FFT conv: (B, H, T)
        xp = x.permute(0, 2, 1)                       # (B, H, T)
        k = self.kernel()                              # (H, T)
        y = _fft_conv(xp, k) + self.D.unsqueeze(-1) * xp   # (B, H, T)
        y = y.permute(0, 2, 1)                         # (B, T, H)
        y = F.gelu(y)
        y = self.mlp(y)
        return self.norm(x + y)


class S4Forecaster(nn.Module):
    """Diagonal S4D-based forecaster (channel-independent)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        d_state: int = 32,
        n_layers: int = 3,
        mlp_mult: int = 2,
        dropout: float = 0.1,
        revin: bool = True,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in
        self._revin_flag = revin

        if revin:
            self.revin_layer = RevIN(enc_in, affine=True)

        self.embed = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList(
            [_S4DBlock(d_model, d_state, seq_len, mlp_mult, dropout)
             for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, pred_len, C)"""
        B, T, C = x.shape

        if self._revin_flag:
            x = self.revin_layer(x, "norm")

        # CI: treat each channel as an independent scalar sequence
        x_ci = x.permute(0, 2, 1).reshape(B * C, T, 1)   # (BC, T, 1)
        x_ci = self.embed(x_ci)                            # (BC, T, d_model)

        for block in self.blocks:
            x_ci = block(x_ci)

        ctx = x_ci.mean(dim=1)                             # (BC, d_model)
        pred = self.head(ctx)                              # (BC, pred_len)
        pred = pred.reshape(B, C, self.pred_len).permute(0, 2, 1)

        if self._revin_flag:
            pred = self.revin_layer(pred, "denorm")
        return pred
