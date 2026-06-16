"""Shared DDPM utilities: noise schedules, forward/reverse process, timestep embeddings."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# ── noise schedules ───────────────────────────────────────────────────────────

def make_beta_schedule(schedule: str, T: int,
                       beta_start: float = 1e-4, beta_end: float = 0.02) -> Tensor:
    """Return beta schedule tensor of length T."""
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, T)
    if schedule == "cosine":
        steps = T + 1
        x = torch.linspace(0, T, steps)
        ac = torch.cos(((x / T) + 0.008) / 1.008 * math.pi / 2) ** 2
        ac = ac / ac[0]
        betas = 1.0 - ac[1:] / ac[:-1]
        return betas.clamp(0.0, 0.999)
    raise ValueError(f"Unknown schedule: {schedule!r}")


# ── sinusoidal timestep embedding ─────────────────────────────────────────────

def sinusoidal_embedding(timesteps: Tensor, dim: int) -> Tensor:
    """Return (B, dim) float32 sinusoidal embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / max(half - 1, 1)
    )
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ── GaussianDiffusion ─────────────────────────────────────────────────────────

class GaussianDiffusion(nn.Module):
    """DDPM core: forward process, reverse sampling, and simplified noise loss."""

    def __init__(self, T: int = 1000, schedule: str = "cosine"):
        super().__init__()
        betas = make_beta_schedule(schedule, T)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, dim=0)          # alphabar_t
        ac_prev = torch.cat([torch.ones(1), ac[:-1]])

        self.T = T
        self.register_buffer("betas", betas)
        self.register_buffer("ac", ac)
        self.register_buffer("ac_prev", ac_prev)
        self.register_buffer("sqrt_ac", ac.sqrt())
        self.register_buffer("sqrt_one_minus_ac", (1.0 - ac).sqrt())
        # posterior variance q(x_{t-1} | x_t, x_0)
        post_var = betas * (1.0 - ac_prev) / (1.0 - ac)
        self.register_buffer("post_var", post_var.clamp(min=1e-20))

    # ── forward process ───────────────────────────────────────────────────────

    def q_sample(self, x0: Tensor, t: Tensor,
                 noise: Optional[Tensor] = None) -> Tensor:
        """x_t ~ q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        s_ac = self.sqrt_ac[t].view(-1, 1, 1)
        s_om = self.sqrt_one_minus_ac[t].view(-1, 1, 1)
        return s_ac * x0 + s_om * noise

    # ── reverse one step ──────────────────────────────────────────────────────

    @torch.no_grad()
    def p_sample(self, denoise_fn, x_t: Tensor, t: int, **kw) -> Tensor:
        t_batch = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)
        eps = denoise_fn(x_t, t_batch, **kw)
        betas_t = self.betas[t_batch].view(-1, 1, 1)
        s_om = self.sqrt_one_minus_ac[t_batch].view(-1, 1, 1)
        sqrt_recip_alpha = (1.0 / (1.0 - betas_t).sqrt())
        mean = sqrt_recip_alpha * (x_t - betas_t / s_om * eps)
        var = self.post_var[t_batch].view(-1, 1, 1)
        noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        return mean + var.sqrt() * noise

    @torch.no_grad()
    def p_sample_loop(self, denoise_fn, shape, device="cpu", **kw) -> Tensor:
        x = torch.randn(shape, device=device)
        for step in reversed(range(self.T)):
            x = self.p_sample(denoise_fn, x, step, **kw)
        return x

    # ── training loss ─────────────────────────────────────────────────────────

    def ddpm_loss(self, denoise_fn, x0: Tensor, **kw) -> Tensor:
        """Simplified DDPM noise-prediction MSE loss."""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        noise_pred = denoise_fn(x_t, t, **kw)
        return nn.functional.mse_loss(noise_pred, noise)
