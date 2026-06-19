"""WaveNet-style dilated causal convolution forecasting model.

Inspired by: van den Oord et al., "WaveNet: A Generative Model for Raw Audio",
arXiv 2016, and its adaptation to time-series forecasting.

Key idea:
  WaveNet uses **dilated causal convolutions** with exponentially increasing
  dilation rates (1, 2, 4, 8, ...) to build an exponentially large receptive
  field while keeping computational cost linear in sequence length.

  Each residual block applies:
    1. Causal dilated conv (dilation = 2^i) → gated activation (tanh + sigmoid).
    2. Residual connection: add conv1×1(gate) back to input.
    3. Skip connection: accumulate conv1×1(gate) into output.

  Multiple **stacks** of L dilation levels give receptive field 2^L per stack.

  The final output is sum of all skip connections → ReLU → 1×1 conv → pred_len.

Architecture:
  - Input projection: (B, T, N) → (B, N, T) → (B*N, d_model, T) causal pad.
  - S stacks × L dilated residual blocks.
  - Output: sum skips → relu → conv1×1 → conv1×1 → (B, pred_len, N).
  - Optional RevIN.

Args:
    seq_len:     input lookback window.
    pred_len:    forecasting horizon.
    enc_in:      number of variates.
    d_model:     base channel size inside residual blocks.
    d_skip:      skip connection channel size (default: d_model).
    kernel_size: dilated convolution kernel size (default 2).
    num_layers:  dilation levels per stack L (default 8 → rates 1..128).
    num_stacks:  number of stacks S (default 1).
    dropout:     dropout in residual blocks.
    revin:       use Reversible Instance Normalisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True)
            self._std = x.std(dim=1, keepdim=True) + self.eps
            x = (x - self._mean) / self._std
            x = x * self.affine_weight + self.affine_bias
        elif mode == "denorm":
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self._std + self._mean
        return x


class _WaveNetBlock(nn.Module):
    """One dilated causal residual block with gated activation + skip."""

    def __init__(self, d_model: int, d_skip: int, kernel_size: int = 2,
                 dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) * dilation   # causal: pad only left

        # Dilated causal convolution — 2× channels for gated activation
        self.conv_dil = nn.Conv1d(
            d_model, 2 * d_model,
            kernel_size=kernel_size, dilation=dilation, padding=padding,
        )
        self.drop = nn.Dropout(dropout)

        # 1×1 projections
        self.conv_res = nn.Conv1d(d_model, d_model, 1)
        self.conv_skip = nn.Conv1d(d_model, d_skip, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``(B*N, d_model, T)``

        Returns:
            residual: ``(B*N, d_model, T)``
            skip:     ``(B*N, d_skip, T)``
        """
        # Causal: remove the extra right-side padding
        h = self.conv_dil(x)
        h = h[:, :, :x.size(2)]    # trim to original T (causal padding)

        # Gated activation: tanh(gate_1) ⊙ sigmoid(gate_2)
        h_tanh = torch.tanh(h[:, :h.size(1) // 2, :])
        h_sig = torch.sigmoid(h[:, h.size(1) // 2:, :])
        gate = self.drop(h_tanh * h_sig)            # (B*N, d_model, T)

        residual = x + self.conv_res(gate)
        skip = self.conv_skip(gate)
        return residual, skip


class WaveNet(nn.Module):
    """WaveNet-style dilated causal conv model for multivariate forecasting."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        d_skip: int = 64,
        kernel_size: int = 2,
        num_layers: int = 8,
        num_stacks: int = 1,
        dropout: float = 0.0,
        revin: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        d_skip = d_skip or d_model

        self.revin = revin
        if revin:
            self.revin_layer = _RevIN(enc_in)

        # Input projection: 1 → d_model
        self.input_proj = nn.Conv1d(1, d_model, 1)

        # Build stacks of dilated residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_stacks):
            for layer in range(num_layers):
                dilation = 2 ** layer
                self.blocks.append(
                    _WaveNetBlock(d_model, d_skip, kernel_size, dilation, dropout)
                )

        # Output head: sum of skips → ReLU → 1×1 → 1×1 → pred_len
        self.out1 = nn.Conv1d(d_skip, d_skip, 1)
        self.out2 = nn.Conv1d(d_skip, pred_len, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, seq_len, enc_in)``

        Returns:
            ``(B, pred_len, enc_in)``
        """
        B, T, N = x.shape

        if self.revin:
            x = self.revin_layer(x, "norm")

        # Channel-independent: (B, T, N) → (B*N, 1, T)
        x_ci = x.permute(0, 2, 1).reshape(B * N, 1, T)
        h = self.input_proj(x_ci)                  # (B*N, d_model, T)

        skip_total = None
        for block in self.blocks:
            h, skip = block(h)
            skip_total = skip if skip_total is None else skip_total + skip

        # Output head: use only the last time step's skip accumulation
        # Actually use mean across time to aggregate the full context
        out = F.relu(skip_total)                   # (B*N, d_skip, T)
        out = F.relu(self.out1(out))               # (B*N, d_skip, T)
        out = self.out2(out)                        # (B*N, pred_len, T)

        # Take the last time step (causal: last position has seen all history)
        out = out[:, :, -1]                        # (B*N, pred_len)
        out = out.reshape(B, N, self.pred_len).permute(0, 2, 1)  # (B, pred_len, N)

        if self.revin:
            out = self.revin_layer(out, "denorm")

        return out
