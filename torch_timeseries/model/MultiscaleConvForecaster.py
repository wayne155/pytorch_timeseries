"""MultiscaleConvForecaster: Inception-style parallel multi-scale temporal CNN.

Key idea:

Instead of building up receptive field sequentially (dilated TCN) or via a
fixed dilation schedule (WaveNet), process the sequence through *parallel*
convolutional branches with different kernel sizes simultaneously, then merge:

    Branch k: GroupNorm → Conv1d(d_ch, d_ch//n_scales, kernel_k, pad) → GELU
    Merged:   Cat(branches, dim=1) → Conv1d(d_ch, d_ch, 1) → residual + LayerNorm

Each branch captures a different temporal scale:
  - Small kernels  (k = 3, 7)  → local, fine-grained patterns
  - Large kernels  (k = 15, 31) → coarse, trend-level patterns

All branches operate in parallel, so every layer simultaneously integrates
information at multiple resolutions — no "waiting" for receptive field to grow.

Architecture comparison:
  WaveNet:                  dilated *causal* gated conv, dilation = 1,2,4,8,…
  TCNForecaster:            stacked dilated *causal* residual blocks.
  TemporalConvAttention:    dilated TCN *then* multi-head attention.
  ModernTCN:                single large kernel per block.
  MultiscaleConvForecaster: *parallel* branches at different scales per block.

Architecture::

    Input (B, T, C)
      → RevIN norm
      → CI reshape → (B·C, 1, T)
      → stem Conv1d(1, d_model, 1)     (channel expand)
      → n_layers × InceptionBlock:
          Pre-GroupNorm
          4 parallel branches → concat(d_model channels)
          Pointwise mix Conv1d(d_model, d_model, 1)
          GELU → residual + LayerNorm
      → global mean-pool over T → (B·C, d_model)
      → Linear head → (B·C, pred_len)
      → reshape → (B, pred_len, C)
      → RevIN denorm

Args:
    seq_len:    input lookback T.
    pred_len:   forecast horizon.
    enc_in:     number of variates C.
    d_model:    total feature width (must be divisible by n_scales).
    n_layers:   number of stacked Inception blocks.
    kernels:    list of kernel sizes for the parallel branches.
    dropout:    dropout probability.
    revin:      use RevIN instance normalisation.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.revin import RevIN

_DEFAULT_KERNELS = (3, 7, 15, 31)


class _InceptionBlock(nn.Module):
    """One multi-scale inception block.

    Input:  (B, d_model, T)
    Output: (B, d_model, T)
    """

    def __init__(self, d_model: int, kernels: Tuple[int, ...], dropout: float) -> None:
        super().__init__()
        n = len(kernels)
        assert d_model % n == 0, "d_model must be divisible by the number of kernel sizes"
        d_branch = d_model // n

        self.pre_norm = nn.GroupNorm(num_groups=min(n, d_model), num_channels=d_model)

        self.branches = nn.ModuleList()
        for k in kernels:
            pad = k // 2
            self.branches.append(
                nn.Conv1d(d_model, d_branch, kernel_size=k, padding=pad, bias=False)
            )

        self.mix = nn.Conv1d(d_model, d_model, kernel_size=1, bias=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_model, T) → (B, d_model, T)"""
        T = x.shape[-1]
        h = self.pre_norm(x)
        # Parallel branches — trim to T in case of even kernels
        outs = [b(h)[..., :T] for b in self.branches]
        h = F.gelu(torch.cat(outs, dim=1))   # (B, d_model, T)
        h = self.mix(h)
        h = self.drop(h)
        # Residual + LayerNorm (LayerNorm on T dimension)
        out = x + h
        # LayerNorm expects (..., d_model): transpose, norm, transpose back
        out = self.norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        return out


class MultiscaleConvForecaster(nn.Module):
    """Inception multi-scale temporal CNN forecaster (channel-independent)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        n_layers: int = 3,
        kernels: Tuple[int, ...] = _DEFAULT_KERNELS,
        dropout: float = 0.1,
        revin: bool = True,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in
        self._revin_flag = revin
        kernels = tuple(kernels)

        if d_model % len(kernels) != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by len(kernels) ({len(kernels)})"
            )

        if revin:
            self.revin_layer = RevIN(enc_in, affine=True)

        # Expand scalar input to d_model channels
        self.stem = nn.Conv1d(1, d_model, kernel_size=1, bias=True)

        self.blocks = nn.ModuleList(
            [_InceptionBlock(d_model, kernels, dropout) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, pred_len, C)"""
        B, T, C = x.shape

        if self._revin_flag:
            x = self.revin_layer(x, "norm")

        # CI: each channel as an independent 1-D sequence
        x_ci = x.permute(0, 2, 1).reshape(B * C, T).unsqueeze(1)   # (BC, 1, T)
        x_ci = self.stem(x_ci)                                      # (BC, d_model, T)

        for block in self.blocks:
            x_ci = block(x_ci)

        ctx = x_ci.mean(dim=-1)                                      # (BC, d_model)
        pred = self.head(ctx)                                        # (BC, pred_len)
        pred = pred.reshape(B, C, self.pred_len).permute(0, 2, 1)

        if self._revin_flag:
            pred = self.revin_layer(pred, "denorm")
        return pred
