"""TSFLow — Flow Matching with Gaussian Process Priors for probabilistic time-series forecasting.

Reference:
    Kollovieh et al. "Flow Matching with Gaussian Process Priors for Probabilistic
    Time Series Forecasting."  NeurIPS 2023 Workshop on Temporal Graph Learning.

Key idea:
    Learn a vector field v_θ(x_t, t, ctx) that transports samples from a GP prior
    (temporally correlated source) to the conditional forecast distribution via an ODE.

    Training — Conditional Flow Matching (CFM) loss:
        L = E‖v_θ(x_t, t, ctx) − (y − x₀)‖²
        where  x_t = (1−t)·x₀ + t·y,  x₀ ~ GP(0, K),  y ~ p(y|x)

    The GP prior K(i,j) = exp(−‖i−j‖²/2l²) provides temporal correlations in the
    source distribution, giving the velocity field a smoother interpolation target
    compared to an i.i.d. Gaussian base.

    Inference — Euler ODE from x₀ ~ GP(0, K) to x₁ ~ p(y|x):
        x_{t+dt} ← x_t + dt · v_θ(x_t, t, ctx)

Architecture:
    1. GRU backbone encodes the lookback window → compact context vector.
    2. A small Transformer velocity network v_θ predicts the instantaneous
       flow direction given (x_t, t, ctx).
    3. During training a GP sample is drawn each step as the source x₀;
       the Cholesky factor is precomputed and cached as a buffer.

Args:
    seq_len:          lookback window length.
    pred_len:         forecasting horizon.
    enc_in:           number of variates.
    d_model:          hidden dimension for encoder and velocity network.
    flow_layers:      Transformer depth of the velocity network (default 3).
    gp_length_scale:  length-scale l of the SE kernel (default 2.0).
    n_steps:          Euler ODE steps at inference (default 20).
    num_samples:      default S for ``sample()`` (default 50).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────── helpers ──────────────────────────────────────


def _sinusoidal_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of scalar flow-time t ∈ [0, 1] → (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    emb = t[:, None] * freqs[None]               # (B, half)
    return torch.cat([emb.sin(), emb.cos()], dim=-1)   # (B, dim)


def _se_kernel(n: int, l: float, device: torch.device) -> torch.Tensor:
    """Squared-Exponential GP kernel matrix K ∈ ℝ^{n×n}."""
    idx = torch.arange(n, dtype=torch.float32, device=device)
    diff = (idx.unsqueeze(0) - idx.unsqueeze(1)) / l    # (n, n)
    return torch.exp(-0.5 * diff ** 2)


# ─────────────────────────── velocity network ─────────────────────────────


class _VelocityTransformer(nn.Module):
    """Predicts the flow velocity v(x_t, t, ctx) → (B, pred_len, enc_in)."""

    def __init__(
        self,
        pred_len: int,
        enc_in: int,
        ctx_dim: int,
        d_model: int,
        n_layers: int,
    ):
        super().__init__()
        n_heads = max(1, d_model // 32)
        self.x_proj   = nn.Linear(enc_in, d_model)
        self.t_mlp    = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.ctx_proj = nn.Linear(ctx_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, enc_in)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_t: (B, pred_len, enc_in)  — noisy forecast at flow-time t
            t:   (B,)                   — flow time ∈ [0, 1]
            ctx: (B, ctx_dim)           — context vector

        Returns:
            velocity: (B, pred_len, enc_in)
        """
        h = self.x_proj(x_t)                           # (B, L, d)
        h = h + self.t_mlp(_sinusoidal_emb(t, h.shape[-1])).unsqueeze(1)
        h = h + self.ctx_proj(ctx).unsqueeze(1)
        h = self.transformer(h)
        return self.out_proj(h)


# ─────────────────────────── main model ───────────────────────────────────


class TSFlowForecaster(nn.Module):
    """Conditional Flow Matching forecaster with a Gaussian Process prior.

    The GP prior replaces the i.i.d. Gaussian source of standard flow
    matching, providing temporal correlations in the noise-to-forecast
    transport map and improving calibration.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        flow_layers: int = 3,
        gp_length_scale: float = 2.0,
        n_steps: int = 20,
        num_samples: int = 50,
    ):
        super().__init__()
        self.pred_len    = pred_len
        self.enc_in      = enc_in
        self.n_steps     = n_steps
        self.num_samples = num_samples

        # GRU context encoder
        self.gru      = nn.GRU(enc_in, d_model, num_layers=2, batch_first=True, dropout=0.1)
        self.pred_head = nn.Linear(d_model, pred_len * enc_in)
        ctx_dim        = d_model

        # Flow velocity network
        self.velocity = _VelocityTransformer(pred_len, enc_in, ctx_dim, d_model, flow_layers)

        # GP Cholesky factor — precomputed, fixed, registered as buffer
        K = _se_kernel(pred_len, gp_length_scale, torch.device("cpu"))
        L = torch.linalg.cholesky(K + 1e-6 * torch.eye(pred_len))
        self.register_buffer("gp_L", L)        # (pred_len, pred_len)

    # ------------------------------------------------------------------ helpers

    def _encode(self, x: torch.Tensor):
        """GRU encode x → (ctx, point_forecast)."""
        _, h_n = self.gru(x)                          # h_n: (n_layers, B, d)
        ctx    = h_n[-1]                               # (B, d_model)
        pred   = self.pred_head(ctx).reshape(-1, self.pred_len, self.enc_in)
        return ctx, pred

    def _gp_sample(self, B: int) -> torch.Tensor:
        """Draw B × enc_in independent GP samples → (B, pred_len, enc_in)."""
        z  = torch.randn(B, self.enc_in, self.pred_len, device=self.gp_L.device)
        x0 = torch.einsum("ij,bcj->bci", self.gp_L, z)   # (B, enc_in, pred_len)
        return x0.permute(0, 2, 1)                         # (B, pred_len, enc_in)

    # ------------------------------------------------------------------ training

    def flow_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Conditional Flow Matching loss.

        Args:
            x: (B, seq_len, enc_in) — lookback
            y: (B, pred_len, enc_in) — ground-truth forecast

        Returns:
            scalar CFM loss.
        """
        B         = x.size(0)
        ctx, _    = self._encode(x)
        x0        = self._gp_sample(B)                    # (B, L, C)
        t         = torch.rand(B, device=x.device)        # (B,)
        t_exp     = t[:, None, None]
        x_t       = (1 - t_exp) * x0 + t_exp * y         # interpolate
        target    = y - x0                                 # straight velocity
        pred_v    = self.velocity(x_t, t, ctx)
        return F.mse_loss(pred_v, target)

    # ------------------------------------------------------------------ forward / sample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Point forecast = GRU backbone projection.

        Args:
            x: (B, seq_len, enc_in)

        Returns:
            (B, pred_len, enc_in)
        """
        _, pred = self._encode(x)
        return pred

    @torch.no_grad()
    def sample(
        self, x: torch.Tensor, num_samples: Optional[int] = None
    ) -> torch.Tensor:
        """Draw samples from p(y|x) via Euler ODE integration.

        Args:
            x:           (B, seq_len, enc_in)
            num_samples: S — defaults to self.num_samples

        Returns:
            (B, pred_len, enc_in, S)
        """
        S      = num_samples or self.num_samples
        B      = x.size(0)
        dt     = 1.0 / self.n_steps
        ctx, _ = self._encode(x)

        samples = []
        for _ in range(S):
            x_t = self._gp_sample(B)
            for step in range(self.n_steps):
                t = torch.full((B,), step * dt, device=x.device)
                x_t = x_t + dt * self.velocity(x_t, t, ctx)
            samples.append(x_t)

        return torch.stack(samples, dim=-1)    # (B, pred_len, enc_in, S)
