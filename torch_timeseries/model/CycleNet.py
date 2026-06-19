"""CycleNet: Explicitly modelling periodic components for time-series forecasting.

Reference: Lin et al., "CycleNet: Enhancing Time Series Forecasting through
Modelling Periodic Patterns", NeurIPS 2024.

Key idea:
  Many real-world time series have strong, stable **periodic patterns**
  (daily, weekly, monthly cycles).  CycleNet decouples these cycles explicitly:

    1. Learn a **shared cycle buffer** C of length `cycle_len` — a learnable
       parameter of shape ``(cycle_len, enc_in)`` that represents one full
       period.
    2. For each window starting at time t, extract the matching segment from
       the cycle buffer and subtract it from the input: r = x − C[t % cycle_len].
       This produces a **residual** series with the periodic component removed.
    3. Apply a simple backbone (default: Linear; optionally MLP) to the residual.
    4. Re-add the future cycle segment: ŷ = backbone(r) + C[(t+T) % cycle_len].

  Because the backbone only needs to model the non-periodic part, a very
  simple linear layer often outperforms complex attention-based models.

Architecture:
  * Cycle buffer: ``nn.Parameter(zeros(cycle_len, enc_in))``.
  * Backbone: either ``Linear(seq_len → pred_len)`` or a two-layer MLP.
  * Optional RevIN before cycle removal.
  * Channel-independent processing.

Args:
    seq_len:    input lookback window T.
    pred_len:   forecasting horizon H.
    enc_in:     number of variates (channels).
    cycle_len:  period length (default 24 — hourly data with daily cycle).
    backbone:   "linear" (default) or "mlp".
    d_model:    hidden size of the MLP backbone (only used when backbone="mlp").
    revin:      use Reversible Instance Normalisation.
    dropout:    dropout in MLP backbone.
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


class CycleNet(nn.Module):
    """CycleNet: learnable periodic cycle buffer + residual backbone."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        cycle_len: int = 24,
        backbone: str = "linear",
        d_model: int = 512,
        revin: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if backbone not in ("linear", "mlp"):
            raise ValueError(f"backbone must be 'linear' or 'mlp', got '{backbone}'")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.cycle_len = cycle_len

        # Learnable cycle: one full period for each variate
        self.cycle = nn.Parameter(torch.zeros(cycle_len, enc_in))

        self.revin = revin
        if revin:
            self.revin_layer = _RevIN(enc_in)

        # Residual backbone (channel-independent)
        if backbone == "linear":
            self.backbone = nn.Linear(seq_len, pred_len)
        else:
            self.backbone = nn.Sequential(
                nn.Linear(seq_len, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, pred_len),
            )

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #

    def _cycle_segment(self, start: int, length: int) -> torch.Tensor:
        """Extract `length` consecutive cycle values starting at position `start`.

        Returns ``(length, enc_in)`` by wrapping around modulo cycle_len.
        """
        idx = torch.arange(start, start + length, device=self.cycle.device) % self.cycle_len
        return self.cycle[idx]                     # (length, enc_in)

    # ------------------------------------------------------------------ #
    # forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor, start_token: int = 0) -> torch.Tensor:
        """
        Args:
            x:           ``(B, seq_len, enc_in)``.
            start_token: time index of the first element in x modulo cycle_len.
                         Defaults to 0 (safe for benchmarks where all windows
                         start at arbitrary positions; using 0 means the cycle
                         removal is approximate but still learns a useful bias).

        Returns:
            ``(B, pred_len, enc_in)``.
        """
        B, T, N = x.shape

        if self.revin:
            x = self.revin_layer(x, "norm")

        # 1. Remove current cycle segment from input
        c_past = self._cycle_segment(start_token, T)    # (T, N)
        residual = x - c_past.unsqueeze(0)              # (B, T, N)

        # 2. Apply backbone (channel-independent)
        #    reshape: (B, T, N) → (B*N, T) → backbone → (B*N, pred_len)
        r_ci = residual.permute(0, 2, 1).reshape(B * N, T)    # (B*N, T)
        pred_r = self.backbone(r_ci)                           # (B*N, pred_len)
        pred_r = pred_r.reshape(B, N, self.pred_len).permute(0, 2, 1)  # (B, pred_len, N)

        # 3. Re-add future cycle segment
        c_future = self._cycle_segment(start_token + T, self.pred_len)  # (pred_len, N)
        pred = pred_r + c_future.unsqueeze(0)           # (B, pred_len, N)

        if self.revin:
            pred = self.revin_layer(pred, "denorm")

        return pred
