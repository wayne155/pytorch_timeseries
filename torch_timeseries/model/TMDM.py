"""TMDM: Temporal diffusion model for unconditional generation (Ye et al.).

Adapts TMDM probabilistic forecasting to unconditional generation.
Uses standard DDPM with a learned prior mean mu(x) and TMDM-style
posterior coefficients for the reverse process.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Shared utilities ──────────────────────────────────────────────────────────

def _extract(vals: Tensor, t: Tensor, ref: Tensor) -> Tensor:
    out = torch.gather(vals.to(t.device), 0, t)
    return out.reshape([t.shape[0]] + [1] * (ref.dim() - 1))


# ── Networks ──────────────────────────────────────────────────────────────────

class _ConditionalLinear(nn.Module):
    """Timestep-conditioned linear: gamma(t) * Linear(x)."""
    def __init__(self, in_dim: int, out_dim: int, T: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.embed = nn.Embedding(T + 1, out_dim)
        self.embed.weight.data.uniform_()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.embed(t).unsqueeze(1) * self.lin(x)


class _TmdmDenoiser(nn.Module):
    """TMDM ConditionalGuidedModel: concat [y_t ‖ y_0_hat] → eps."""
    def __init__(self, n_features: int, T: int, hidden: int = 128):
        super().__init__()
        self.l1 = _ConditionalLinear(n_features * 2, hidden, T)
        self.l2 = _ConditionalLinear(hidden, hidden, T)
        self.l3 = _ConditionalLinear(hidden, hidden, T)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, y_t: Tensor, y_0_hat: Tensor, t: Tensor) -> Tensor:
        h = torch.cat([y_t, y_0_hat], dim=-1)
        h = F.softplus(self.l1(h, t))
        h = F.softplus(self.l2(h, t))
        h = F.softplus(self.l3(h, t))
        return self.out(h)


class _MuNet(nn.Module):
    """GRU-based prior mean predictor: (B, T, N) → (B, T, N)."""
    def __init__(self, n_features: int, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=n_layers, batch_first=True)
        self.proj = nn.Linear(hidden, n_features)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.gru(x)
        return self.proj(out)


# ── TMDM ─────────────────────────────────────────────────────────────────────

class TMDM(nn.Module):
    """Temporal diffusion model for unconditional time series generation.

    Adapts TMDM (Ye) from probabilistic forecasting to generation using
    a GRU prior-mean network and TMDM-style DDPM reverse coefficients.

    Args:
        seq_len: window length T
        n_features: number of channels C
        T: diffusion timesteps (default 100)
        beta_start: initial beta (default 1e-4)
        beta_end: final beta (default 0.5)
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        T: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.5,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.T = T

        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(0)
        alpha_bar_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_bar_sqrt", alphas_cumprod.sqrt())
        self.register_buffer("one_minus_alphas_bar_sqrt", (1.0 - alphas_cumprod).sqrt())
        self.register_buffer("alphas_cumprod_prev", alpha_bar_prev)

        self.mu_net = _MuNet(n_features)
        self.denoiser = _TmdmDenoiser(n_features, T)

    def _q_sample(self, x: Tensor, y_0_hat: Tensor, t: Tensor, e: Tensor) -> Tensor:
        """TMDM forward: x_t = sqrt(a_bar)·x + (1−sqrt(a_bar))·y_0_hat + sqrt(1−a_bar)·e."""
        a_bar = _extract(self.alphas_bar_sqrt, t, x)
        one_minus = _extract(self.one_minus_alphas_bar_sqrt, t, x)
        return a_bar * x + (1 - a_bar) * y_0_hat + one_minus * e

    def loss(self, x: Tensor) -> Tensor:
        """TMDM training loss on a batch of windows (B, T, N)."""
        B = x.shape[0]
        device = x.device

        y_0_hat = self.mu_net(x)

        t = torch.randint(0, self.T, (B // 2 + 1,), device=device)
        t = torch.cat([t, self.T - 1 - t])[:B]

        e = torch.randn_like(x)
        x_t = self._q_sample(x, y_0_hat, t, e)

        eps_pred = self.denoiser(x_t, y_0_hat, t)
        loss_eps = (e - eps_pred).square().mean()
        loss_mu = (y_0_hat - x).square().mean()
        return loss_eps + loss_mu

    @torch.no_grad()
    def generate(self, n: int, device: str = "cpu") -> Tensor:
        """Reverse-diffusion sampling; returns (n, seq_len, n_features) on CPU."""
        dev = torch.device(device)
        x_dummy = torch.zeros(n, self.seq_len, self.n_features, device=dev)
        y_0_hat = self.mu_net(x_dummy)

        # Prior at T: N(y_0_hat, I)
        cur = torch.randn_like(y_0_hat) + y_0_hat

        for step in reversed(range(1, self.T)):
            t = torch.full((n,), step, dtype=torch.long, device=dev)
            eps_pred = self.denoiser(cur, y_0_hat, t)

            at = _extract(self.alphas, t, cur)
            one_minus = _extract(self.one_minus_alphas_bar_sqrt, t, cur)
            a_bar = _extract(self.alphas_bar_sqrt, t, cur)
            one_minus_prev_sq = 1.0 - _extract(self.alphas_cumprod_prev, t, cur)
            a_bar_m1_sqrt = _extract(self.alphas_cumprod_prev, t, cur).sqrt()

            # y_0 reparameterization (TMDM style)
            y_0_rep = (cur - (1 - a_bar) * y_0_hat - eps_pred * one_minus) / a_bar.clamp(min=1e-8)

            # Posterior mean coefficients (from TMDM p_sample)
            one_m_sq = one_minus.square()
            g0 = (1 - at) * a_bar_m1_sqrt / one_m_sq
            g1 = one_minus_prev_sq * at.sqrt() / one_m_sq
            g2 = 1 + (a_bar - 1) * (at.sqrt() + a_bar_m1_sqrt) / one_m_sq

            y_mean = g0 * y_0_rep + g1 * cur + g2 * y_0_hat
            beta_hat = one_minus_prev_sq / one_m_sq * (1 - at)
            cur = y_mean + beta_hat.clamp(min=0).sqrt() * torch.randn_like(cur)

        # Final step t=0
        t0 = torch.zeros(n, dtype=torch.long, device=dev)
        a_bar0 = _extract(self.alphas_bar_sqrt, t0, cur)
        one_minus0 = _extract(self.one_minus_alphas_bar_sqrt, t0, cur)
        eps0 = self.denoiser(cur, y_0_hat, t0)
        x0 = (cur - (1 - a_bar0) * y_0_hat - eps0 * one_minus0) / a_bar0.clamp(min=1e-8)

        return x0.cpu()
