"""MinGRUForecaster — Minimal GRU (minGRU) recurrent forecaster.

minGRU strips the standard GRU to its essential gating, removing the reset gate
and the tanh non-linearity on the hidden state:

    z_t = σ(W_z · x_t + b_z)               # forget gate (no h-dependence)
    h̃_t = W_h · x_t + b_h                  # candidate (linear, no tanh)
    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

Key properties:
  - No tanh on h → h is a linear weighted average of past inputs
  - No reset gate → one fewer parameter matrix
  - Parallelisable via log-sum-exp prefix scan (we use sequential scan here)
  - Still expressive through stacking and the linear projection W_h

Reference: Yang et al., "Were RNNs All We Needed?" (2024) arXiv:2410.01201
"""

from __future__ import annotations
import torch
import torch.nn as nn


class _MinGRULayer(nn.Module):
    """One MinGRU layer: no tanh, no reset gate, sequential scan."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.W_z = nn.Linear(d_model, d_model, bias=True)   # gate
        self.W_h = nn.Linear(d_model, d_model, bias=True)   # candidate

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, d)
        B, T, D = x_seq.shape
        h = torch.zeros(B, D, device=x_seq.device, dtype=x_seq.dtype)
        hiddens = []
        for t in range(T):
            z = torch.sigmoid(self.W_z(x_seq[:, t]))       # (B, d)
            h_tilde = self.W_h(x_seq[:, t])                # (B, d) — no tanh
            h = (1 - z) * h + z * h_tilde
            hiddens.append(h)
        return torch.stack(hiddens, dim=1)                  # (B, T, d)


class _MinGRUBlock(nn.Module):
    """MinGRU layer + GELU MLP + LayerNorm + residual."""

    def __init__(self, d_model: int, d_ffn: int) -> None:
        super().__init__()
        self.gru = _MinGRULayer(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        x = self.norm1(x + self.gru(x))
        x = self.norm2(x + self.ffn(x))
        return x


class MinGRUForecaster(nn.Module):
    """MinGRU multivariate forecaster.

    Channel-independent (CI) + RevIN.

    Args:
        seq_len:    input sequence length
        pred_len:   forecast horizon
        enc_in:     number of input channels
        d_model:    hidden state dimension
        d_ffn:      feedforward hidden dimension (default 4×d_model)
        n_layers:   number of stacked MinGRU blocks
        dropout:    dropout on head
        revin:      apply RevIN instance normalisation
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        d_ffn: int | None = None,
        n_layers: int = 2,
        dropout: float = 0.1,
        revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.revin = revin
        d_ffn = d_ffn or d_model * 4

        if revin:
            from torch_timeseries.nn.revin import RevIN
            self.rev = RevIN(enc_in)

        self.embed = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList(
            [_MinGRUBlock(d_model, d_ffn) for _ in range(n_layers)]
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        if self.revin:
            x = self.rev(x, "norm")

        B, T, C = x.shape
        x_ci = x.permute(0, 2, 1).reshape(B * C, T, 1)
        h = self.embed(x_ci)                    # (BC, T, d)

        for block in self.blocks:
            h = block(h)

        out = self.head(self.drop(h[:, -1, :]))     # last hidden → (BC, pred_len)
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)

        if self.revin:
            out = self.rev(out, "denorm")
        return out
