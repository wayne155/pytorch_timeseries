"""HGRN2Forecaster — Hierarchical Gated Recurrent Network v2 forecaster.

HGRN2 (Qin et al., 2024) uses a *norm-preserving* recurrence:

    f_t = sigmoid(W_f · x_t)      — forget gate (input-only, no h dependence)
    i_t = sqrt(1 − f_t²)          — input gate (algebraically coupled to f)
    u_t = tanh(W_u · x_t)        — candidate update
    h_t = f_t ⊙ h_{t-1} + i_t ⊙ u_t
    y_t = σ(W_o · x_t) ⊙ h_t    — output gate

Key property: ||h_t|| ≤ 1 is maintained by construction because
  f_t² + i_t² = f_t² + (1 − f_t²) = 1,
so the update is a rotation-like operation, not merely a convex combination.

Architecturally distinct from:
  MinGRUForecaster:  no tanh candidate, no sqrt coupling, no output gate
  GRU/LSTM:          forget & input gates depend on h_{t-1} (sequential conv)
  QRNNForecaster:    gates from causal conv over a window, not point-wise linear
  RWKVForecaster:    WKV max-normalised recurrence, token-shift mixing

Reference: Qin et al., "HGRN2: Gated Linear RNNs with State Expansion", 2024.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _HGRN2Cell(nn.Module):
    """Single HGRN2 recurrent layer."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.W_f = nn.Linear(d_model, d_model, bias=True)
        self.W_u = nn.Linear(d_model, d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, d_model)
        B, T, D = x_seq.shape
        h = torch.zeros(B, D, device=x_seq.device, dtype=x_seq.dtype)
        hs = []
        for t in range(T):
            x_t = x_seq[:, t]
            f_t = torch.sigmoid(self.W_f(x_t))
            i_t = torch.sqrt(torch.clamp(1.0 - f_t * f_t, min=1e-6))
            u_t = torch.tanh(self.W_u(x_t))
            h = f_t * h + i_t * u_t
            o_t = torch.sigmoid(self.W_o(x_t))
            hs.append(o_t * h)
        return torch.stack(hs, dim=1)   # (B, T, D)


class _HGRN2Block(nn.Module):
    """HGRN2Cell + GELU-FFN + pre-norm + residual."""

    def __init__(self, d_model: int, d_ffn: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.cell = _HGRN2Cell(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.drop(self.cell(self.norm1(x)))
        return h + self.ffn(self.norm2(h))


class HGRN2Forecaster(nn.Module):
    """HGRN2 multivariate forecaster with channel-independent (CI) + RevIN.

    Args:
        seq_len:   input sequence length
        pred_len:  forecast horizon
        enc_in:    number of input channels
        d_model:   hidden dimension
        d_ffn:     feedforward hidden size
        n_layers:  number of HGRN2 blocks
        dropout:   dropout rate
        revin:     apply RevIN instance normalisation
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        d_ffn: int = 256,
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
        self.blocks = nn.ModuleList([
            _HGRN2Block(d_model, d_ffn, dropout)
            for _ in range(n_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin:
            x = self.rev(x, "norm")

        B, T, C = x.shape
        x_ci = x.permute(0, 2, 1).reshape(B * C, T, 1)
        h = self.embed(x_ci)            # (BC, T, d)

        for block in self.blocks:
            h = block(h)

        out = self.head(self.drop(h[:, -1]))    # (BC, pred_len)
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)

        if self.revin:
            out = self.rev(out, "denorm")
        return out
