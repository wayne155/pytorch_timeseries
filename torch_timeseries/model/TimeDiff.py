"""TimeDiff: self-guidance diffusion for time series (Shen & Kwok, 2023).

Key addition over standard DDPM: *future mixup* during training provides
the model with a noisy self-guidance signal by concatenating a lower-noise
version of x0 with x_t as input to the denoising network.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .diffusion_utils import GaussianDiffusion, sinusoidal_embedding


class _TimeDiffDenoiser(nn.Module):
    def __init__(self, seq_len: int, n_features: int,
                 d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        # Input: concat [x_t, guidance] along feature dim
        self.in_proj = nn.Linear(n_features * 2, d_model)
        self.t_proj  = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 2, batch_first=True, norm_first=False
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, n_layers)
        self.out_proj = nn.Linear(d_model, n_features)

    def forward(self, x_t: Tensor, t: Tensor, guidance: Tensor) -> Tensor:
        # x_t, guidance: (B, T, C)
        h = self.in_proj(torch.cat([x_t, guidance], dim=-1))   # (B, T, D)
        D = h.shape[-1]
        t_emb = self.t_proj(sinusoidal_embedding(t, D))        # (B, D)
        h = h + t_emb[:, None, :]
        return self.out_proj(self.encoder(h))                  # (B, T, C)


class TimeDiff(nn.Module):
    """TimeDiff model.

    Args:
        seq_len: sequence length
        n_features: number of channels
        d_model: transformer hidden dim (default 128)
        n_heads: attention heads (default 4)
        n_layers: encoder layers (default 4)
        T: diffusion steps (default 500)
        schedule: "linear" (default) or "cosine"
        mix_ratio: future-mixup strength; fraction of t used for guidance (default 0.5)
    """

    def __init__(self, seq_len: int, n_features: int,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 4,
                 T: int = 500, schedule: str = "linear", mix_ratio: float = 0.5):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.mix_ratio = mix_ratio
        self.diffusion = GaussianDiffusion(T=T, schedule=schedule)
        self.net = _TimeDiffDenoiser(seq_len, n_features, d_model, n_heads, n_layers)

    def denoise(self, x_t: Tensor, t: Tensor, guidance: Tensor = None) -> Tensor:
        if guidance is None:
            guidance = torch.zeros_like(x_t)
        return self.net(x_t, t, guidance)

    def loss(self, x0: Tensor) -> Tensor:
        """DDPM loss with future-mixup guidance."""
        B = x0.shape[0]
        t = torch.randint(0, self.diffusion.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.diffusion.q_sample(x0, t, noise)
        # guidance: noisy version of x0 at a lower noise level
        t_guide = (t.float() * self.mix_ratio).long().clamp(0, self.diffusion.T - 1)
        guidance = self.diffusion.q_sample(x0, t_guide)
        noise_pred = self.net(x_t, t, guidance)
        return nn.functional.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def generate(self, n: int, device: str = "cpu") -> Tensor:
        shape = (n, self.seq_len, self.n_features)
        return self.diffusion.p_sample_loop(
            self.denoise, shape, device=device
        ).cpu()
