"""LiquidNetForecaster — Liquid Time-Constant (LTC) recurrent forecaster.

Each LTC cell solves a first-order ODE:
    dh/dt = -h/τ(x,h) + σ(Wx·x + Wh·h)
    τ(x,h) = softplus(Wτx·x + Wτh·h)         # data-dependent time constant
    h_new ≈ h + Δt · dh/dt                    # explicit Euler integration
    h_new = h + Δt·σ(Wx·x+Wh·h) / (1+Δt/τ)  # semi-implicit stabilized form

Reference: Hasani et al., "Liquid Time-constant Networks" (2021)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _LTCCell(nn.Module):
    """Single Liquid Time-Constant cell (semi-implicit Euler)."""

    def __init__(self, d_model: int, dt: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.dt = dt
        # Input → state drive
        self.W_x = nn.Linear(d_model, d_model, bias=True)
        self.W_h = nn.Linear(d_model, d_model, bias=False)
        # Time-constant τ (positive via softplus)
        self.W_tau_x = nn.Linear(d_model, d_model, bias=True)
        self.W_tau_h = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x, h: (batch, d_model)
        drive = torch.tanh(self.W_x(x) + self.W_h(h))
        tau = F.softplus(self.W_tau_x(x) + self.W_tau_h(h)) + 1e-3
        # semi-implicit: h_new = (h + dt·drive) / (1 + dt/τ)
        h_new = (h + self.dt * drive) / (1.0 + self.dt / tau)
        return h_new


class _LTCBlock(nn.Module):
    """LTC cell wrapped with residual projection and LayerNorm."""

    def __init__(self, d_model: int, dt: float) -> None:
        super().__init__()
        self.cell = _LTCCell(d_model, dt)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (batch, T, d_model)
        B, T, D = x_seq.shape
        h = torch.zeros(B, D, device=x_seq.device, dtype=x_seq.dtype)
        hiddens = []
        for t in range(T):
            h = self.cell(x_seq[:, t], h)
            hiddens.append(h)
        out = torch.stack(hiddens, dim=1)   # (B, T, D)
        return self.norm(out + x_seq)


class LiquidNetForecaster(nn.Module):
    """Multi-layer Liquid Time-Constant network for multivariate forecasting.

    Channel-independent (CI) strategy: each channel processed separately.

    Args:
        seq_len:    input sequence length
        pred_len:   forecast horizon
        enc_in:     number of input channels
        d_model:    hidden state dimension
        n_layers:   number of stacked LTC blocks
        dt:         Euler integration step size
        dropout:    dropout rate on head
        revin:      apply RevIN instance normalisation
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        n_layers: int = 2,
        dt: float = 0.1,
        dropout: float = 0.1,
        revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.revin = revin

        if revin:
            from torch_timeseries.nn.revin import RevIN
            self.rev = RevIN(enc_in)

        self.embed = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([_LTCBlock(d_model, dt) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        if self.revin:
            x = self.rev(x, "norm")

        B, T, C = x.shape
        # CI: (B*C, T, 1) → embed → (B*C, T, d_model)
        x_ci = x.permute(0, 2, 1).reshape(B * C, T, 1)
        h = self.embed(x_ci)                   # (BC, T, d_model)

        for block in self.blocks:
            h = block(h)                        # (BC, T, d_model)

        h_last = h[:, -1, :]                    # (BC, d_model) — last state
        out = self.head(self.drop(h_last))      # (BC, pred_len)
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)  # (B, pred_len, C)

        if self.revin:
            out = self.rev(out, "denorm")
        return out
