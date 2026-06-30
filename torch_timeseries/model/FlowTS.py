"""FlowTS — Time Series Generation via Rectified Flow.

Reference:
    Hu et al. "FlowTS: Time Series Generation via Rectified Flow."
    ICML 2024 / arXiv 2024.

Key idea:
    Rectified flow (Liu et al. 2022) learns a straight-line mapping between
    noise x₀ ~ N(0,I) and data x₁ ~ p(x) by training a velocity network
    v_θ(x_t, t) to match the linear interpolant velocity (x₁ − x₀):

        L = E‖v_θ(x_t, t) − (x₁ − x₀)‖²
        where  x_t = (1−t)·x₀ + t·x₁,  t ~ U[0, 1]

    At generation time the learned ODE dx/dt = v_θ(x_t, t) is integrated
    from x₀ ~ N(0,I) with Euler steps to produce synthetic time series.

    Unlike score-based diffusion (CSDI, DiffusionTS) the model works in
    data space directly (no forward noising process), and the straight-line
    paths make Euler integration highly efficient (fewer steps needed).

Architecture:
    A Transformer-based velocity network conditioned on flow time t via
    sinusoidal embeddings attends jointly over all time steps and channels,
    then projects back to the original feature dimension.

Args:
    seq_len:    window length of each generated sequence.
    n_features: number of channels / variates.
    d_model:    Transformer hidden dimension (default 64).
    n_heads:    number of attention heads (default 4).
    n_layers:   Transformer depth (default 4).
    n_steps:    Euler ODE steps at generation time (default 20).

Tasks: Generation.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────── helpers ──────────────────────────────────────


def _sinusoidal_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of scalar t ∈ [0, 1] → (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    emb = t[:, None] * freqs[None]
    return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ─────────────────────────── velocity network ─────────────────────────────


class _RectifiedVelocityNet(nn.Module):
    """Transformer velocity network v_θ(x_t, t) → dx/dt."""

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
    ):
        super().__init__()
        self.x_proj   = nn.Linear(n_features, d_model)
        self.t_mlp    = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.pos_emb  = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, n_features)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: (B, T, C) — noisy sequence at flow-time t
            t:   (B,)       — flow time ∈ [0, 1]

        Returns:
            velocity: (B, T, C)
        """
        B, T, _ = x_t.shape
        h = self.x_proj(x_t)                                         # (B, T, d)
        pos = torch.arange(T, device=x_t.device)
        h = h + self.pos_emb(pos).unsqueeze(0)                       # (B, T, d)
        h = h + self.t_mlp(_sinusoidal_emb(t, h.shape[-1])).unsqueeze(1)  # broadcast t
        h = self.transformer(h)                                       # (B, T, d)
        return self.out_proj(h)                                       # (B, T, C)


# ─────────────────────────── main model ───────────────────────────────────


class FlowTS(nn.Module):
    """Rectified Flow generative model for time series.

    Learns straight-line transport from Gaussian noise to the data
    distribution, enabling high-quality generation with very few ODE steps.
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        n_steps: int = 20,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.n_features = n_features
        self.n_steps    = n_steps

        self.velocity = _RectifiedVelocityNet(seq_len, n_features, d_model, n_heads, n_layers)

    # ------------------------------------------------------------------ training

    def loss(self, x1: torch.Tensor) -> torch.Tensor:
        """Rectified flow training loss.

        Args:
            x1: (B, seq_len, n_features) — real time-series windows.

        Returns:
            scalar MSE between predicted and target velocity.
        """
        B   = x1.size(0)
        x0  = torch.randn_like(x1)                        # noise sample
        t   = torch.rand(B, device=x1.device)             # uniform flow time
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1   # interpolate
        target = x1 - x0                                   # straight velocity
        return F.mse_loss(self.velocity(x_t, t), target)

    # ------------------------------------------------------------------ generation

    @torch.no_grad()
    def generate(
        self,
        n: int,
        device: str = "cpu",
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate n synthetic time-series windows via Euler ODE integration.

        Args:
            n:       number of sequences to generate.
            device:  torch device string.
            n_steps: override the number of Euler steps (default: self.n_steps).

        Returns:
            (n, seq_len, n_features) on CPU.
        """
        steps = n_steps or self.n_steps
        dt    = 1.0 / steps
        x     = torch.randn(n, self.seq_len, self.n_features, device=device)

        for step in range(steps):
            t = torch.full((n,), step * dt, device=device)
            x = x + dt * self.velocity(x, t)

        return x.cpu()
