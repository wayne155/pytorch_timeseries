"""NS-Diffusion: Non-stationary adaptive diffusion for time series (Ye et al.).

Architecture hook points for the author:
* ``_adaptive_beta``: returns a (B,) beta value; replace with your paper's formulation.
* ``_NSDenoiser.forward``: Transformer backbone with RevIN.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .diffusion_utils import sinusoidal_embedding


# ── Reversible Instance Normalization (RevIN) ─────────────────────────────────

class RevIN(nn.Module):
    """Reversible instance normalization (Kim et al., 2022)."""

    def __init__(self, n_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(n_features))
        self.affine_bias   = nn.Parameter(torch.zeros(n_features))

    def normalize(self, x: Tensor) -> Tensor:
        # x: (B, T, C)
        self._mean = x.mean(dim=1, keepdim=True)          # (B, 1, C)
        self._std  = x.std(dim=1, keepdim=True) + self.eps
        x = (x - self._mean) / self._std
        return x * self.affine_weight + self.affine_bias

    def denormalize(self, x: Tensor) -> Tensor:
        x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        return x * self._std + self._mean


# ── Adaptive beta schedule MLP ────────────────────────────────────────────────

class _AdaptiveScheduleMLP(nn.Module):
    """Maps [t_emb, mu, sigma] -> beta_t in (1e-4, 0.02)."""

    def __init__(self, t_emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(t_emb_dim + 2, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        self._lo, self._hi = 1e-4, 0.02

    def forward(self, t_emb: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        # t_emb: (B, D), mu/sigma: (B,)
        inp = torch.cat([t_emb, mu[:, None], sigma[:, None]], dim=-1)
        raw = self.net(inp).squeeze(-1)   # (B,) in (0,1)
        return self._lo + (self._hi - self._lo) * raw


# ── Denoising network ─────────────────────────────────────────────────────────

class _NSDenoiser(nn.Module):
    def __init__(self, seq_len: int, n_features: int,
                 d_model: int, n_heads: int, n_layers: int, t_emb_dim: int):
        super().__init__()
        self.revin   = RevIN(n_features)
        self.in_proj = nn.Linear(n_features, d_model)
        self.t_proj  = nn.Sequential(
            nn.Linear(t_emb_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 2, batch_first=True, norm_first=False
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, n_layers)
        self.out_proj = nn.Linear(d_model, n_features)

    def forward(self, x_t: Tensor, t: Tensor, t_emb: Tensor) -> Tensor:
        x_norm = self.revin.normalize(x_t)
        h = self.in_proj(x_norm) + self.t_proj(t_emb)[:, None, :]
        out = self.out_proj(self.encoder(h))     # (B, T, C) — noise in normed space
        return self.revin.denormalize(out)


# ── NSDiffusion ───────────────────────────────────────────────────────────────

class NSDiffusion(nn.Module):
    """Non-stationary adaptive diffusion model.

    Args:
        seq_len: sequence length
        n_features: number of channels
        d_model: transformer hidden dim (default 128)
        n_heads: attention heads (default 4)
        n_layers: encoder depth (default 4)
        T: diffusion steps (default 500)
        t_emb_dim: sinusoidal timestep embedding dim (default 64)
    """

    def __init__(self, seq_len: int, n_features: int,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 4,
                 T: int = 500, t_emb_dim: int = 64):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.T = T
        self.t_emb_dim = t_emb_dim

        self.schedule_mlp = _AdaptiveScheduleMLP(t_emb_dim)
        self.denoiser = _NSDenoiser(seq_len, n_features, d_model, n_heads, n_layers, t_emb_dim)

    def _adaptive_beta(self, t: Tensor, x0: Tensor) -> Tensor:
        """Per-sequence adaptive beta_t.  (B,) in (1e-4, 0.02)."""
        t_emb = sinusoidal_embedding(t, self.t_emb_dim)
        mu    = x0.mean(dim=(1, 2))          # (B,)
        sigma = x0.std(dim=(1, 2)) + 1e-5
        return self.schedule_mlp(t_emb, mu, sigma)

    def _q_sample(self, x0: Tensor, t: Tensor) -> tuple:
        """Forward process with adaptive schedule. Returns (x_t, noise)."""
        betas = self._adaptive_beta(t, x0).view(-1, 1, 1)   # (B,1,1)
        alpha = 1.0 - betas
        noise = torch.randn_like(x0)
        x_t = alpha.sqrt() * x0 + (1.0 - alpha).sqrt() * noise
        return x_t, noise

    def denoise(self, x_t: Tensor, t: Tensor) -> Tensor:
        t_emb = sinusoidal_embedding(t, self.t_emb_dim)
        return self.denoiser(x_t, t, t_emb)

    def loss(self, x0: Tensor) -> Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_t, noise = self._q_sample(x0, t)
        noise_pred = self.denoise(x_t, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def generate(self, n: int, device: str = "cpu") -> Tensor:
        """Ancestral sampling with fixed mean beta at each step."""
        x = torch.randn(n, self.seq_len, self.n_features, device=device)
        mean_beta = 0.02 / self.T
        for step in reversed(range(self.T)):
            t_batch = torch.full((n,), step, dtype=torch.long, device=device)
            betas = torch.full((n, 1, 1), mean_beta, device=device)
            alpha = 1.0 - betas
            eps = self.denoise(x, t_batch)
            mean = (x - (1.0 - alpha).sqrt() * eps) / alpha.sqrt()
            noise = torch.randn_like(x) if step > 0 else torch.zeros_like(x)
            x = mean + betas.sqrt() * noise
        return x.cpu()
