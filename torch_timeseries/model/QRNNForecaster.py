"""QRNNForecaster — Quasi-Recurrent Neural Network for time-series forecasting.

QRNN (Bradbury et al., 2016) separates the computation into two stages:

1. **Convolutional stage** (fully parallel, no h dependence):
      z_t = tanh(W_z ⋆ x_{t-k:t})   — candidate update
      f_t = σ(W_f ⋆ x_{t-k:t})      — forget gate
      o_t = σ(W_o ⋆ x_{t-k:t})      — output gate

2. **Pooling stage** (sequential, but trivially simple):
      h_t = f_t ⊙ h_{t-1} + (1 − f_t) ⊙ z_t   (fo-pooling)
      y_t = o_t ⊙ h_t

This is architecturally distinct from:
  - TCNForecaster: causal dilated conv with no gating, no recurrent pooling
  - GRU/LSTM: gates depend on previous hidden state h_{t-1} (sequential even in gate computation)
  - ConformerForecaster: depthwise conv inside a Transformer-like structure

The decoupling of gate computation from recurrence enables the gates to be
computed at full speed (batched convolution), with only a lightweight scan
for the pooling step.

Reference: Bradbury et al., "Quasi-Recurrent Neural Networks", ICLR 2017.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _QRNNLayer(nn.Module):
    """Single QRNN layer: causal conv gates + fo-pool + output gating."""

    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for symmetric padding"
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.pad_len = kernel_size - 1  # causal left-padding

        # One conv producing 3×d_model channels: [z, f, o]
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=3 * d_model,
            kernel_size=kernel_size,
            padding=0,   # we handle causal padding manually
        )
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, D = x.shape

        # Causal convolution: left-pad so output length == input length
        x_t = x.permute(0, 2, 1)                          # (B, D, T)
        x_pad = F.pad(x_t, (self.pad_len, 0))              # (B, D, T+pad)
        out = self.conv(x_pad)                             # (B, 3D, T)
        out = out.permute(0, 2, 1)                         # (B, T, 3D)

        Z_raw, F_raw, O_raw = out.chunk(3, dim=-1)         # each (B, T, D)
        Z = torch.tanh(Z_raw)
        F_gate = torch.sigmoid(F_raw)
        O = torch.sigmoid(O_raw)

        # fo-pooling: h_t = f_t * h_{t-1} + (1-f_t) * z_t
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        hs = []
        for t in range(T):
            h = F_gate[:, t] * h + (1.0 - F_gate[:, t]) * Z[:, t]
            hs.append(h)
        H = torch.stack(hs, dim=1)   # (B, T, D)

        # Output gate
        Y = O * H                    # (B, T, D)
        Y = self.drop(Y)
        return self.norm(Y + x)      # residual connection


class _QRNNBlock(nn.Module):
    """QRNNLayer + GELU-FFN + pre-norm + residual."""

    def __init__(self, d_model: int, d_ffn: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.qrnn = _QRNNLayer(d_model, kernel_size, dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qrnn(x)
        return x + self.ffn(x)


class QRNNForecaster(nn.Module):
    """QRNN multivariate forecaster with channel-independent (CI) + RevIN.

    Args:
        seq_len:     input sequence length
        pred_len:    forecast horizon
        enc_in:      number of input channels
        d_model:     hidden dimension
        d_ffn:       feedforward hidden size
        n_layers:    number of QRNN blocks
        kernel_size: causal convolution kernel size (must be odd)
        dropout:     dropout rate
        revin:       apply RevIN instance normalisation
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        d_ffn: int = 256,
        n_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        revin: bool = True,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.revin = revin

        if revin:
            from torch_timeseries.nn.revin import RevIN
            self.rev = RevIN(enc_in)

        self.embed = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            _QRNNBlock(d_model, d_ffn, kernel_size, dropout)
            for _ in range(n_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin:
            x = self.rev(x, "norm")

        B, T, C = x.shape
        x_ci = x.permute(0, 2, 1).reshape(B * C, T, 1)
        h = self.embed(x_ci)                         # (BC, T, d)

        for block in self.blocks:
            h = block(h)

        out = self.head(self.drop(h[:, -1]))          # last step → (BC, pred_len)
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)

        if self.revin:
            out = self.rev(out, "denorm")
        return out
