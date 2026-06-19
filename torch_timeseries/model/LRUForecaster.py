"""LRUForecaster: Linear Recurrent Unit for time series forecasting.

Key idea (Orvieto et al., 2023 — "Resurrecting Recurrent Neural Networks for
Long Sequences"):

Replace the dense state-transition matrix of a classic RNN with a *complex
diagonal* matrix whose eigenvalues are explicitly constrained to the open
unit disk:

    Λ_i = sigmoid(ν_i) · exp(i · π · tanh(θ_i))

This gives:
    |Λ_i| = sigmoid(ν_i) ∈ (0, 1)       → guaranteed stability
    arg(Λ_i) = π · tanh(θ_i) ∈ (−π, π) → learnable frequency

The recurrence is then element-wise on the complex hidden state h_t:

    h_t = Λ ⊙ h_{t-1} + B x_t           (complex)
    y_t = Re(C h_t) + D x_t             (real output)

Because Λ is diagonal, the d_state recurrences are independent scalar
complex multiplications — no matrix–matrix products inside the loop.

Architecture::

    Input (B, T, C)
      → RevIN norm
      → CI reshape → (B·C, T)
      → Linear embed → (B·C, T, d_model)
      → n_layers × [ LRU cell → GELU MLP → LayerNorm + residual ]
      → global mean-pool → (B·C, d_model)
      → Linear head → (B·C, pred_len)
      → reshape → (B, pred_len, C)
      → RevIN denorm

Architecture comparison:
  RNNForecaster:   vanilla GRU / LSTM — dense non-linear state transitions.
  MambaForecaster: selective SSM (S6) — input-dependent state selection gate.
  BiLSTMForecaster: bidirectional LSTM with Bahdanau attention.
  LRUForecaster:   diagonal complex linear recurrence — no gates, no attention.

Args:
    seq_len:   input lookback T.
    pred_len:  forecast horizon.
    enc_in:    number of variates C.
    d_model:   model (embedding) width.
    d_state:   complex hidden-state dimension per LRU cell.
    n_layers:  number of stacked LRU + MLP blocks.
    mlp_mult:  expansion factor for the pointwise MLP inside each block.
    dropout:   dropout probability.
    revin:     use RevIN instance normalisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.revin import RevIN


class _LRUCell(nn.Module):
    """Single linear recurrent unit cell.

    Processes (B, T, d_model) with complex diagonal recurrence of size
    d_state, then projects back to d_model via the real part of C h_t.
    """

    def __init__(self, d_model: int, d_state: int, dropout: float) -> None:
        super().__init__()
        self.d_state = d_state

        # Eigenvalue parameterisation: |Λ| = sigmoid(ν), arg = π·tanh(θ)
        self.nu = nn.Parameter(torch.zeros(d_state))
        self.theta = nn.Parameter(torch.zeros(d_state))

        # Input projection: ℝ^d_model → ℂ^d_state
        self.B_re = nn.Parameter(torch.randn(d_state, d_model) * 0.02)
        self.B_im = nn.Parameter(torch.randn(d_state, d_model) * 0.02)

        # Output projection: ℂ^d_state → ℝ^d_model  (take real part of C h)
        self.C_re = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.C_im = nn.Parameter(torch.randn(d_model, d_state) * 0.02)

        # Skip connection D (real, per d_model dimension)
        self.D = nn.Parameter(torch.ones(d_model))

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        B, T, _ = x.shape

        # Build eigenvalues (d_state,) — computed once per forward pass
        lam_abs = torch.sigmoid(self.nu)
        lam_arg = torch.pi * torch.tanh(self.theta)
        lam_re = lam_abs * torch.cos(lam_arg)   # (d_state,)
        lam_im = lam_abs * torch.sin(lam_arg)   # (d_state,)

        # Pre-project input to complex state space: (B, T, d_state) × 2
        bx_re = x @ self.B_re.T   # (B, T, d_state)
        bx_im = x @ self.B_im.T

        # Sequential scan over time
        h_re = x.new_zeros(B, self.d_state)
        h_im = x.new_zeros(B, self.d_state)
        outs = []
        for t in range(T):
            # h = Λ ⊙ h + B x_t  (complex element-wise)
            new_re = lam_re * h_re - lam_im * h_im + bx_re[:, t]
            new_im = lam_re * h_im + lam_im * h_re + bx_im[:, t]
            h_re, h_im = new_re, new_im
            # y_t = Re(C h_t)  =  C_re h_re - C_im h_im
            y_t = h_re @ self.C_re.T - h_im @ self.C_im.T   # (B, d_model)
            outs.append(y_t)

        y = torch.stack(outs, dim=1)    # (B, T, d_model)
        y = y + self.D * x              # skip
        return self.norm(x + self.drop(y))


class _LRUBlock(nn.Module):
    """LRU cell followed by a pointwise GELU MLP."""

    def __init__(self, d_model: int, d_state: int, mlp_mult: int, dropout: float) -> None:
        super().__init__()
        self.lru = _LRUCell(d_model, d_state, dropout)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * mlp_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lru(x)
        return x + self.mlp(x)


class LRUForecaster(nn.Module):
    """Stacked Linear Recurrent Unit forecaster (channel-independent)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        d_state: int = 64,
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
            [_LRUBlock(d_model, d_state, mlp_mult, dropout) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, pred_len, C)"""
        B, T, C = x.shape

        if self._revin_flag:
            x = self.revin_layer(x, "norm")

        # CI: each channel as an independent sequence of scalars
        x_ci = x.permute(0, 2, 1).reshape(B * C, T, 1)   # (BC, T, 1)
        x_ci = self.embed(x_ci)                            # (BC, T, d_model)

        for block in self.blocks:
            x_ci = block(x_ci)

        # Global mean-pool over time → head
        ctx = x_ci.mean(dim=1)                             # (BC, d_model)
        pred = self.head(ctx)                              # (BC, pred_len)
        pred = pred.reshape(B, C, self.pred_len).permute(0, 2, 1)  # (B, pred_len, C)

        if self._revin_flag:
            pred = self.revin_layer(pred, "denorm")
        return pred
