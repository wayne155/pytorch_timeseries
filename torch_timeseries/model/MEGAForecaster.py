"""MEGAForecaster — Moving-Average Equipped Gated Attention forecaster.

Core idea: replace dot-product attention's K/Q interaction with a Multi-dimensional
Damped EMA (MD-EMA) that gives each hidden dimension its own learned timescale:

    h_t = β ⊙ x_t + (1-β) ⊙ h_{t-1}     # EMA mixing
    β = sigmoid(β_raw)                      # per-dim decay in (0,1)

EMA outputs feed single-head gated attention (GSAH):
    u = W_u · x,   z = W_z · x            # value and gate projections
    q = W_q · ema, k = W_k · ema          # queries/keys from EMA states
    attn_w = softmax(q·k^T / sqrt(d_h))
    ctx = attn_w · (z ⊙ sigmoid(u))        # gated value
    out = W_o · ctx                        # project back

Reference: Ma et al., "Mega: Moving Average Equipped Gated Attention" (2022)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _EMALayer(nn.Module):
    """Multi-dimensional Damped EMA: per-dim learnable decay β ∈ (0,1)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        # Unconstrained parameter; β = sigmoid(β_raw)
        self.beta_raw = nn.Parameter(torch.zeros(d_model))
        # Trainable input mixing coefficient (η in MEGA paper)
        self.eta_raw = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        B, T, D = x.shape
        beta = torch.sigmoid(self.beta_raw)          # (d,)
        eta = torch.sigmoid(self.eta_raw)            # (d,) scale input contribution

        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            h = beta * h + (1 - beta) * (eta * x[:, t])
            outputs.append(h)
        return torch.stack(outputs, dim=1)           # (B, T, d)


class _GSAHBlock(nn.Module):
    """Gated Single-head Attention + EMA block (MEGA block).

    EMA provides keys & queries; original input provides gated values.
    """

    def __init__(self, d_model: int, d_expand: int, dropout: float) -> None:
        super().__init__()
        self.scale = math.sqrt(d_expand)
        self.ema = _EMALayer(d_model)
        # Project d_model → d_expand for queries/keys
        self.W_q = nn.Linear(d_model, d_expand, bias=True)
        self.W_k = nn.Linear(d_model, d_expand, bias=True)
        # Value and gate from original input
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        self.W_gate = nn.Linear(d_model, d_model, bias=True)
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        ema_out = self.ema(x)                               # (B, T, d)

        Q = self.W_q(ema_out)                               # (B, T, d_exp)
        K = self.W_k(ema_out)                               # (B, T, d_exp)
        V = self.W_v(x)                                     # (B, T, d)
        gate = torch.sigmoid(self.W_gate(x))                # (B, T, d)

        # Causal single-head attention over EMA states
        attn = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, T, T)
        causal_mask = torch.triu(
            torch.full((Q.size(1), K.size(1)), float("-inf"), device=x.device),
            diagonal=1
        )
        attn = F.softmax(attn + causal_mask, dim=-1)

        # Gated value
        gated_v = gate * V                                  # (B, T, d)
        ctx = torch.bmm(self.drop(attn), gated_v)           # (B, T, d)
        out = self.drop(self.W_o(ctx))

        return self.norm(out + x)


class MEGAForecaster(nn.Module):
    """MEGA multivariate forecaster.

    Channel-independent (CI) + RevIN.

    Args:
        seq_len:    input sequence length
        pred_len:   forecast horizon
        enc_in:     number of input channels
        d_model:    model dimension
        d_expand:   QK projection dimension for attention
        n_layers:   number of MEGA blocks
        dropout:    dropout rate
        revin:      apply RevIN instance normalisation
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        d_expand: int = 32,
        n_layers: int = 2,
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
        self.blocks = nn.ModuleList(
            [_GSAHBlock(d_model, d_expand, dropout) for _ in range(n_layers)]
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin:
            x = self.rev(x, "norm")

        B, T, C = x.shape
        x_ci = x.permute(0, 2, 1).reshape(B * C, T, 1)
        h = self.embed(x_ci)                            # (BC, T, d)

        for block in self.blocks:
            h = block(h)

        out = self.head(self.drop(h.mean(dim=1)))       # (BC, pred_len)
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)

        if self.revin:
            out = self.rev(out, "denorm")
        return out
