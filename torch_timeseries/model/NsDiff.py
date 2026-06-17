"""NS-Diffusion: Non-stationary diffusion for unconditional generation (Ye et al.).

Adapts NsDiff probabilistic forecasting model to unconditional generation.
Key idea: forward/reverse diffusion noise is scaled by learned local variance gx,
preserving non-stationary structure without requiring an external conditioning signal.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Schedule utilities ────────────────────────────────────────────────────────

def _make_beta_schedule(T: int, beta_start: float, beta_end: float) -> Tensor:
    return torch.linspace(beta_start, beta_end, T)


def _compute_tilde_alpha(alpha: Tensor) -> Tensor:
    """tilde_alpha[t] = sum_{s=0}^{t} prod_{u=s}^{t} alpha[u]  (NS-DDPM paper eq.)"""
    n = alpha.shape[0]
    result = torch.zeros(n, dtype=alpha.dtype, device=alpha.device)
    for t in range(n):
        sl = alpha[: t + 1].flip(0)          # [a_t, a_{t-1}, ..., a_0]
        result[t] = torch.cumprod(sl, 0).sum()
    return result


def _compute_hat_alpha(alpha: Tensor) -> Tensor:
    n = alpha.shape[0]
    result = torch.zeros(n, dtype=alpha.dtype, device=alpha.device)
    for t in range(n):
        sl = alpha[: t + 1].flip(0)
        cp = torch.cumprod(sl, 0)
        result[t] = (cp * sl).sum()
    return result


def _extract(vals: Tensor, t: Tensor, ref: Tensor) -> Tensor:
    """Gather scalar schedule values at timestep t, broadcast to ref shape."""
    out = torch.gather(vals.to(t.device), 0, t)
    return out.reshape([t.shape[0]] + [1] * (ref.dim() - 1))


# ── Local variance estimation ─────────────────────────────────────────────────

def _wv_sigma_trailing(x: Tensor, window: int) -> Tensor:
    """Trailing rolling variance: (B, T, N) → (B, T, N), same shape as input."""
    w = min(window, x.shape[1])
    x_pad = F.pad(x, (0, 0, w - 1, 0), mode="replicate")
    wins = x_pad.unfold(1, w, 1)                        # (B, T, N, w)
    return wins.var(dim=-1, unbiased=False).clamp(min=1e-8)


# ── Networks ──────────────────────────────────────────────────────────────────

class _NsDenoiser(nn.Module):
    """GRU-based conditional denoiser: [y_t ‖ y_0_hat ‖ gx] → (eps, sigma).

    Uses a bidirectional GRU so each position is denoised with full temporal
    context, enabling the model to produce smooth, correlated sequences.
    """
    def __init__(self, n_features: int, T: int, hidden: int = 64):
        super().__init__()
        self.inp_proj  = nn.Linear(n_features * 3, hidden)
        self.t_embed   = nn.Embedding(T + 1, hidden)
        self.gru       = nn.GRU(hidden, hidden, num_layers=2,
                                batch_first=True, bidirectional=True)
        self.eps_head   = nn.Linear(hidden * 2, n_features)
        self.sigma_head = nn.Linear(hidden * 2, n_features)

    def forward(self, y_t: Tensor, y_0_hat: Tensor, gx: Tensor, t: Tensor):
        h = torch.cat([y_t, y_0_hat, gx], dim=-1)          # (B, T, 3N)
        h = self.inp_proj(h) + self.t_embed(t).unsqueeze(1) # (B, T, hidden)
        h, _ = self.gru(h)                                   # (B, T, 2·hidden)
        return self.eps_head(h), F.softplus(self.sigma_head(h))


class _SigmaNet(nn.Module):
    """Estimate per-position local variance gx from the input window.

    Computes trailing rolling variance, discards the padded prefix, then maps
    (B, N, T−k) → (B, T, N) via MLP + softplus so output is always > 0.
    """
    def __init__(self, seq_len: int, n_features: int, hidden: int = 64, kernel_size: int = 24):
        super().__init__()
        k = min(kernel_size, seq_len - 1)
        self.k = k
        in_len = seq_len - k
        self.mlp = nn.Sequential(
            nn.Linear(in_len, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, seq_len),
        )

    def forward(self, x: Tensor) -> Tensor:
        k = self.k
        # x.unfold(1, k, 1) → (B, T-k+1, N, k); take first T-k windows
        wins = x.unfold(1, k, 1)                              # (B, T-k+1, N, k)
        sigma = wins.var(dim=-1, unbiased=False)[:, :-1, :]   # (B, T-k, N)
        sigma = sigma.clamp(min=1e-8)
        out = F.softplus(self.mlp(sigma.permute(0, 2, 1)))    # (B, N, T)
        return out.permute(0, 2, 1)                           # (B, T, N)


# ── NsDiff ───────────────────────────────────────────────────────────────────

class NsDiff(nn.Module):
    """NsDiff — Non-Stationary Diffusion for time series generation (Ye et al.).

    Extends DDPM by scaling the forward and reverse noise with a per-step local
    variance estimate ``gx``, so the diffusion process is non-stationary and
    adapts to heteroscedastic sequences.  The denoiser is a bidirectional GRU
    that models the full temporal structure at each diffusion step.  At generation
    time ``gx`` is estimated from a reservoir of training windows accumulated
    during ``loss()`` calls.

    Args:
        seq_len (int): Sequence length of each window.
        n_features (int): Number of channels.
        T (int): Number of diffusion steps. Defaults to 100.
        beta_start (float): Starting noise schedule value. Defaults to 1e-4.
        beta_end (float): Ending noise schedule value. Defaults to 0.01.
        kernel_size (int): Rolling-window size for local variance estimation.
            Defaults to 24.

    Tasks: Generation.
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        T: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.01,
        kernel_size: int = 24,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.T = T
        self.kernel_size = min(kernel_size, seq_len - 1)

        betas = _make_beta_schedule(T, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(0)
        tilde_alpha = _compute_tilde_alpha(alphas)
        hat_alpha = _compute_hat_alpha(alphas)
        betas_tilde = (tilde_alpha - hat_alpha).clamp(min=0.0)
        betas_bar = 1.0 - alphas_cumprod

        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_bar_sqrt", alphas_cumprod.sqrt())
        self.register_buffer("one_minus_alphas_bar_sqrt", (1.0 - alphas_cumprod).sqrt())
        self.register_buffer("alphas_cumprod_prev",
                             torch.cat([torch.ones(1), alphas_cumprod[:-1]]))
        self.register_buffer("betas_tilde", betas_tilde)
        self.register_buffer("betas_tilde_m_1",
                             torch.cat([torch.ones(1), betas_tilde[:-1]]))
        self.register_buffer("betas_bar", betas_bar)
        self.register_buffer("betas_bar_m_1",
                             torch.cat([torch.ones(1), betas_bar[:-1]]))

        self.sigma_net = _SigmaNet(seq_len, n_features, kernel_size=self.kernel_size)
        self.denoiser = _NsDenoiser(n_features, T)

        # Reservoir of training windows; sampled at generation time to obtain
        # realistic, diverse gx values without needing the caller to pass data.
        self.register_buffer("_x_bank", torch.zeros(0, seq_len, n_features))
        self._bank_size = 512

    # ── forward-process helpers ───────────────────────────────────────────────

    def _fwd_noise_var(self, gx: Tensor, y_sigma: Tensor, t: Tensor) -> Tensor:
        """Noise variance: (b_bar − b_tilde)·gx + b_tilde·y_sigma  ≥ 0."""
        b_bar = _extract(self.betas_bar, t, gx)
        b_tilde = _extract(self.betas_tilde, t, gx)
        return ((b_bar - b_tilde) * gx + b_tilde * y_sigma).clamp(min=1e-8)

    def _sigma_tilde(self, gx: Tensor, y_sigma: Tensor, t: Tensor) -> Tensor:
        """Posterior variance used in KL loss."""
        at = _extract(self.alphas, t, gx)
        S1 = (1 - at) ** 2 * gx + at * (1 - at) * y_sigma
        btm1 = _extract(self.betas_bar_m_1, t, gx)
        btl_m1 = _extract(self.betas_tilde_m_1, t, gx)
        S2 = (btm1 - btl_m1) * gx + btl_m1 * y_sigma
        return (S1 * S2 / (at * S2 + S1).clamp(min=1e-8)).clamp(min=1e-8)

    def _q_sample(self, x: Tensor, gx: Tensor, y_sigma: Tensor, t: Tensor, e: Tensor) -> Tensor:
        """Forward process: x_t = sqrt(a_bar)·x + (1−sqrt(a_bar))·0 + e·sqrt(noise_var)."""
        a_bar = _extract(self.alphas_bar_sqrt, t, x)
        noise_var = self._fwd_noise_var(gx, y_sigma, t)
        # y_0_hat = 0 (normalized data has near-zero mean)
        return a_bar * x + e * noise_var.sqrt()

    # ── training loss ─────────────────────────────────────────────────────────

    def loss(self, x: Tensor) -> Tensor:
        """NS-DDPM loss on a batch of windows (B, T, N)."""
        B = x.shape[0]
        device = x.device

        y_sigma = _wv_sigma_trailing(x, self.kernel_size)
        gx = self.sigma_net(x)

        # Reservoir-sample training windows so generate() can draw realistic gx
        with torch.no_grad():
            bank = torch.cat([self._x_bank.detach().cpu(), x.detach().cpu()], dim=0)
            if bank.shape[0] > self._bank_size:
                idx = torch.randperm(bank.shape[0])[:self._bank_size]
                bank = bank[idx]
            self._x_bank = bank

        y_0_hat = torch.zeros_like(x)

        t = torch.randint(0, self.T, (B // 2 + 1,), device=device)
        t = torch.cat([t, self.T - 1 - t])[:B]

        e = torch.randn_like(x)
        x_t = self._q_sample(x, gx, y_sigma, t, e)

        sigma_tilde = self._sigma_tilde(gx, y_sigma, t)
        eps_pred, sigma_theta = self.denoiser(x_t, y_0_hat, gx, t)
        sigma_theta = sigma_theta + 1e-8

        loss_eps = (e - eps_pred).square().mean()
        ratio = sigma_tilde / sigma_theta
        loss_kl = ratio.mean() - torch.log(ratio + 1e-8).mean()
        loss_gx = (gx.sqrt() - y_sigma.sqrt()).square().mean()
        return loss_eps + loss_kl + loss_gx

    # ── generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, n: int, device: str = "cpu", x_ref: Tensor | None = None) -> Tensor:
        """Reverse-diffusion sampling; returns (n, seq_len, n_features) on CPU.

        Args:
            n: number of sequences to generate.
            device: torch device string.
            x_ref: optional reference windows (B, T, C) used to estimate the local
                variance gx.  When None, gx defaults to all-ones (unit variance),
                matching standardised training data.
        """
        dev = torch.device(device)
        if x_ref is not None:
            x_ref = x_ref.to(dev)
            reps = (n + x_ref.shape[0] - 1) // x_ref.shape[0]
            x_ref = x_ref.repeat(reps, 1, 1)[:n]
            gx = self.sigma_net(x_ref)
        elif self._x_bank.shape[0] > 0:
            # Sample n windows from the training reservoir to get diverse, realistic gx
            bank = self._x_bank
            idx = torch.randperm(bank.shape[0])[:n]
            x_ref_bank = bank[idx].to(dev)
            if x_ref_bank.shape[0] < n:
                x_ref_bank = x_ref_bank.repeat((n // x_ref_bank.shape[0]) + 1, 1, 1)[:n]
            gx = self.sigma_net(x_ref_bank)
        else:
            gx = torch.ones(n, self.seq_len, self.n_features, device=dev)
        y_0_hat = torch.zeros_like(gx)

        # Prior at T: N(0, gx)
        cur = gx.sqrt() * torch.randn_like(gx)

        for step in reversed(range(1, self.T)):
            t = torch.full((n,), step, dtype=torch.long, device=dev)
            eps_pred, _ = self.denoiser(cur, y_0_hat, gx, t)

            at = _extract(self.alphas, t, cur)
            a_bar = _extract(self.alphas_bar_sqrt, t, cur)
            noise_var = (self.betas_bar[step] * gx).clamp(min=1e-8)   # y_sigma≈gx

            # y_0 reparameterization
            y_0_rep = (cur - eps_pred * noise_var.sqrt()) / a_bar.clamp(min=1e-8)

            # Posterior mean: gamma coefficients (y_sigma = gx simplification)
            a_bar_m1 = _extract(self.alphas_cumprod_prev, t, cur).sqrt()
            btm1 = _extract(self.betas_bar_m_1, t, cur)
            btl_m1 = _extract(self.betas_tilde_m_1, t, cur)
            S1 = (1 - at) * gx
            S2 = btm1 * gx
            denom = (at * S2 + S1).clamp(min=1e-8)
            g0 = a_bar_m1 * S1 / denom
            g1 = at.sqrt() * S2 / denom
            g2 = (at.sqrt() * (at - 1) * S2 + (1 - a_bar_m1) * S1) / denom

            y_mean = g0 * y_0_rep + g1 * cur + g2 * y_0_hat
            post_var = (btl_m1 * gx).clamp(min=1e-8)
            cur = y_mean + post_var.sqrt() * torch.randn_like(cur)

        # Final step t=0
        t0 = torch.zeros(n, dtype=torch.long, device=dev)
        a_bar0 = _extract(self.alphas_bar_sqrt, t0, cur)
        noise_var0 = (self.betas_bar[0] * gx).clamp(min=1e-8)
        eps0, _ = self.denoiser(cur, y_0_hat, gx, t0)
        x0 = (cur - eps0 * noise_var0.sqrt()) / a_bar0.clamp(min=1e-8)

        return x0.cpu()
