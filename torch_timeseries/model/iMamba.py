"""iMamba: Inverted Mamba for multivariate time-series forecasting.

Motivation:
  iTransformer (ICLR 2024) demonstrated that treating each variate as a single
  token — and then applying attention across variates — is surprisingly powerful
  for multivariate forecasting.  The key insight is that the "sequence" that the
  Transformer sees is the channel dimension, not the time dimension.

  iMamba replaces iTransformer's quadratic-complexity self-attention with a
  linear-complexity Selective SSM (Mamba-1 style).  This enables:
    • O(C) instead of O(C²) cross-channel modelling (useful when C is large)
    • Inherently ordered processing of variates (the order is determined by
      the channel-correlation ranking pre-sorting step, or simply left as is)
    • Better preservation of long-range variate dependencies via the
      recurrent state

  Architecture:
    1. Each variate's time series x_c ∈ R^T is projected to a d_model embedding
       (the "variate token").  This forms a sequence of C tokens for the SSM.
    2. L Mamba blocks process the C-length variate-token sequence.
       Each block: expand → SSM + gate → contract, with LayerNorm + residual.
    3. Each variate token is independently projected to pred_len via a shared
       linear head.
    4. RevIN normalisation.

  Note: Unlike S-Mamba (which uses Mamba for intra-variate temporal patterns
  and Transformer for inter-variate), iMamba uses only Mamba but on the
  *inverted* variate-as-sequence axis.

Args:
    seq_len:     input lookback length.
    pred_len:    forecast horizon.
    enc_in:      number of variates (= C).
    d_model:     variate token dimension.
    d_state:     SSM state dimension.
    e_layers:    number of Mamba blocks.
    d_ff:        optional feed-forward hidden size (0 = skip FF layer).
    dropout:     dropout rate.
    revin:       use RevIN instance normalisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# RevIN
# ──────────────────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────────────────
# Selective SSM
# ──────────────────────────────────────────────────────────────────────────────


class _SelectiveSSM(nn.Module):
    """Diagonal input-dependent SSM with ZOH discretisation (Mamba-1 core)."""

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.W_delta = nn.Linear(d_model, d_model, bias=True)
        self.W_B = nn.Linear(d_model, d_state, bias=False)
        self.W_C = nn.Linear(d_model, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        A = -torch.exp(self.A_log)
        delta = F.softplus(self.W_delta(x))
        Bx = self.W_B(x)
        Cx = self.W_C(x)
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            dA = torch.exp(delta[:, t, :].unsqueeze(-1) * A.unsqueeze(0))
            dB = delta[:, t, :].unsqueeze(-1) * Bx[:, t, :].unsqueeze(1)
            h = dA * h + dB * x[:, t, :].unsqueeze(-1)
            y_t = (Cx[:, t, :].unsqueeze(1) * h).sum(-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        return y + self.D * x


class _MambaBlock(nn.Module):
    """Mamba block with expand-SSM-gate-contract + residual."""

    def __init__(self, d_model: int, d_state: int, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.ssm = _SelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        z, gate = self.in_proj(x).chunk(2, dim=-1)
        z = self.ssm(z)
        h = z * F.silu(gate)
        return residual + self.drop(self.out_proj(h))


# ──────────────────────────────────────────────────────────────────────────────
# iMamba
# ──────────────────────────────────────────────────────────────────────────────


class iMamba(nn.Module):
    """Inverted Mamba: SSM over the variate (channel) axis."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        d_state: int = 16,
        e_layers: int = 3,
        d_ff: int = 256,
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

        # Embed each variate's time series T → d_model (one token per variate)
        self.embed = nn.Linear(seq_len, d_model)

        # Mamba blocks process the C-length variate-token sequence
        self.mamba_layers = nn.ModuleList(
            [_MambaBlock(d_model, d_state, dropout=dropout) for _ in range(e_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

        # Output head: d_model → pred_len per variate token
        self.head = nn.Linear(d_model, pred_len)

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

        # Invert: each variate becomes a token
        # (B, T, C) → (B, C, T) → embed T → d_model
        x_var = x.transpose(1, 2)               # (B, C, T)
        h = self.embed(x_var)                   # (B, C, d_model)

        # Process C-length sequence with Mamba
        for layer in self.mamba_layers:
            h = layer(h)                        # (B, C, d_model)
        h = self.final_norm(h)

        # Decode: each variate token → pred_len
        out = self.head(h)                      # (B, C, pred_len)
        out = out.transpose(1, 2)               # (B, pred_len, C)

        if self._revin_flag:
            out = self.revin_layer(out, "denorm")
        return out
