"""MambaForecaster: Selective State Space Model for time-series forecasting.

References:
  Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
  ICLR 2024.
  Wang et al., "Is Mamba Effective for Time Series Forecasting?", 2024.

Key ideas:
  State Space Models (SSMs) generalise RNNs with a structured recurrence:
      h_t = A * h_{t-1} + B * x_t      (state transition)
      y_t = C * h_t                     (output)
  where A ∈ R^{N×N}, B ∈ R^{N×1}, C ∈ R^{1×N} for a single channel.

  S4 uses a fixed (non-input-dependent) diagonal A with HiPPO initialisation.

  **Mamba (Selective SSM / S6)** makes B, C, and the step-size Δ
  *input-dependent*, allowing the model to selectively remember or forget
  based on the content of the current input.  This is the key distinction
  from S4 and all previous SSMs.

  For each position t, Mamba computes:
      Δ_t = softplus(W_Δ * x_t)          (input-dependent step size)
      B_t = W_B * x_t                     (input-dependent B)
      C_t = W_C * x_t                     (input-dependent C)

  The discrete-time matrices are obtained via ZOH discretisation:
      A_bar = exp(Δ_t * A)               (diagonal A → elementwise exp)
      B_bar = (A_bar - I) * A^{-1} * B_t ≈ Δ_t * B_t   (for HiPPO-N)

  The selective scan over T steps produces outputs h_1 … h_T.

  For efficiency this implementation uses a *parallel associative scan*
  instead of a sequential loop (same result, O(T log T) parallelism).

Simplified architecture:
  - Channel-independent Mamba blocks (one SSM per variate, similar to DLinear).
  - Each block: linear expand → SSM → gate → contract.
  - Stack of `e_layers` Mamba blocks.
  - RevIN normalisation.
  - Output: direct linear from last-position state to pred_len.

Args:
    seq_len:   input lookback window.
    pred_len:  forecast horizon.
    enc_in:    number of variates.
    d_model:   expanded channel dimension inside each Mamba block.
    d_state:   SSM state size N.
    e_layers:  number of stacked Mamba blocks.
    dropout:   dropout rate.
    revin:     use RevIN instance normalisation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SelectiveSSM(nn.Module):
    """Single selective state space layer (S6 core).

    Processes a sequence (B, T, d_model) → (B, T, d_model) using a diagonal
    SSM with input-dependent B, C, Δ.  Runs as a sequential scan (correct and
    simple; use associative scan for large T in production).
    """

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Input projections for B, C, Δ
        self.W_B = nn.Linear(d_model, d_state, bias=False)
        self.W_C = nn.Linear(d_model, d_state, bias=False)
        self.W_delta = nn.Linear(d_model, d_model, bias=True)

        # Diagonal A (log-parameterised for stability): shape (d_model, d_state)
        A_log = torch.arange(1, d_state + 1, dtype=torch.float32).log()
        A_log = A_log.unsqueeze(0).expand(d_model, -1)  # (d_model, d_state)
        self.A_log = nn.Parameter(A_log.clone())

        # D (skip connection scalar, per channel)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        B, T, D = x.shape
        N = self.d_state

        # Input-dependent parameters
        delta = F.softplus(self.W_delta(x))   # (B, T, D) — step size > 0
        Bx = self.W_B(x)                      # (B, T, N)
        Cx = self.W_C(x)                      # (B, T, N)

        # Discrete A (ZOH approximation): A_bar = exp(Δ * A)
        # A: (D, N) → neg (stable), delta: (B, T, D)
        A = -torch.exp(self.A_log)            # (D, N)  — negative for stability
        # delta: (B, T, D), A: (D, N) → (B, T, D, N)
        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, T, D, N)

        # Discrete B: dB = delta * B_t  (B, T, D, N)
        dB = delta.unsqueeze(-1) * Bx.unsqueeze(2)  # (B, T, D, N)

        # Sequential selective scan
        # h: (B, D, N)
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            # h ← A_bar_t * h + B_bar_t * x_t
            # dA_t: (B, D, N), h: (B, D, N), dB_t: (B, D, N), x_t: (B, D)
            h = dA[:, t, :, :] * h + dB[:, t, :, :] * x[:, t, :].unsqueeze(-1)
            # y_t = C_t @ h_t (per channel): (B, D)
            y_t = (Cx[:, t, :].unsqueeze(1) * h).sum(-1)  # (B, D)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)            # (B, T, D)
        return y + self.D * x                 # skip connection


class _MambaBlock(nn.Module):
    """Mamba block: expand → SSM + gating → contract, with skip."""

    def __init__(self, d_input: int, d_model: int, d_state: int, dropout: float):
        super().__init__()
        self.expand = nn.Linear(d_input, d_model * 2)
        self.ssm = _SelectiveSSM(d_model, d_state)
        self.contract = nn.Linear(d_model, d_input)
        self.norm = nn.LayerNorm(d_input)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_input)
        z = self.expand(x)                          # (B, T, 2*d_model)
        z1, gate = z.chunk(2, dim=-1)               # each (B, T, d_model)
        z1 = self.ssm(z1)                           # SSM branch
        gate = F.silu(gate)                         # gating branch
        out = z1 * gate                             # elementwise gate
        out = F.dropout(self.contract(out), p=self.dropout, training=self.training)
        return self.norm(x + out)                   # residual


class _RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._mean = x.mean(1, keepdim=True)
            self._std = x.std(1, keepdim=True).clamp(self.eps)
            x = (x - self._mean) / self._std
            return x * self.affine_weight + self.affine_bias
        else:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            return x * self._std + self._mean


class MambaForecaster(nn.Module):
    """Mamba-based time series forecaster.

    Stack of selective SSM (Mamba) blocks with RevIN normalisation.
    Channel-mixing via the full d_model dimension across all variates.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        d_state: int = 16,
        e_layers: int = 2,
        dropout: float = 0.05,
        revin: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self._revin_flag = revin
        if revin:
            self.revin_layer = _RevIN(enc_in)

        # Project enc_in → d_model for channel mixing
        self.input_proj = nn.Linear(enc_in, d_model)

        self.blocks = nn.ModuleList(
            [_MambaBlock(d_model, d_model, d_state, dropout) for _ in range(e_layers)]
        )

        # Decode: last-position hidden → pred_len * enc_in
        self.output_proj = nn.Linear(d_model, pred_len * enc_in)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)
        Returns:
            (B, pred_len, C)
        """
        B, T, C = x.shape

        if self._revin_flag:
            x = self.revin_layer(x, "norm")

        h = self.input_proj(x)              # (B, T, d_model)
        h = F.dropout(h, p=self.dropout, training=self.training)

        for block in self.blocks:
            h = block(h)

        # Decode from last position
        out = self.output_proj(h[:, -1, :])  # (B, pred_len * C)
        out = out.reshape(B, self.pred_len, C)

        if self._revin_flag:
            out = self.revin_layer(out, "denorm")
        return out
