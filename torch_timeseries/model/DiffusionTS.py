"""Diffusion-TS: DDPM with trend+seasonal decomposition (Yuan & Qiao, 2024)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .diffusion_utils import GaussianDiffusion, sinusoidal_embedding


class _DiffusionTSDenoiser(nn.Module):
    """Transformer decoder with separate trend and seasonal output heads."""

    def __init__(self, seq_len: int, n_features: int,
                 d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.in_proj = nn.Linear(n_features, d_model)
        self.t_proj  = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_model * 2, batch_first=True, norm_first=False
        )
        self.decoder = nn.TransformerDecoder(dec_layer, n_layers)
        # Learnable query tokens: one per time step
        self.query = nn.Parameter(torch.randn(1, seq_len, d_model))
        # Trend head: smooth linear projection over the time axis
        self.trend_head  = nn.Linear(seq_len, seq_len)
        self.trend_proj  = nn.Linear(d_model, n_features)
        # Seasonal head: residual (high-frequency) component
        self.seasonal_proj = nn.Linear(d_model, n_features)

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        B, T, C = x_t.shape
        mem = self.in_proj(x_t)                            # (B, T, D)
        D = mem.shape[-1]
        t_emb = self.t_proj(sinusoidal_embedding(t, D))   # (B, D)
        mem = mem + t_emb[:, None, :]
        query = self.query.expand(B, -1, -1)               # (B, T, D)
        out = self.decoder(query, mem)                     # (B, T, D)
        # Trend: smooth projection along time
        trend = self.trend_proj(
            self.trend_head(out.permute(0, 2, 1)).permute(0, 2, 1)
        )                                                  # (B, T, C)
        seasonal = self.seasonal_proj(out)                 # (B, T, C)
        return trend + seasonal                            # (B, T, C)


class DiffusionTS(nn.Module):
    """Diffusion-TS — DDPM with trend-seasonal decomposition (Yuan & Qiao, ICLR 2024).

    Denoiser is a Transformer decoder that separately learns trend and seasonal
    components at each diffusion step.  The seasonal branch uses multi-head
    attention over the sequence; the trend branch uses a simple linear layer.
    Uses a cosine beta schedule by default.

    Paper: *Diffusion-TS: Interpretable Diffusion for General Time Series
    Generation.*
    https://openreview.net/forum?id=4h1apFjO99

    Args:
        seq_len (int): Sequence length of each window.
        n_features (int): Number of channels.
        d_model (int): Transformer hidden dimension. Defaults to 128.
        n_heads (int): Number of attention heads. Defaults to 4.
        n_layers (int): Number of Transformer decoder layers. Defaults to 4.
        T (int): Number of diffusion steps. Defaults to 1000.
        schedule (str): Beta schedule — ``'cosine'`` or ``'linear'``.
            Defaults to ``'cosine'``.

    Tasks: Generation.
    """

    def __init__(self, seq_len: int, n_features: int,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 4,
                 T: int = 1000, schedule: str = "cosine"):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.diffusion = GaussianDiffusion(T=T, schedule=schedule)
        self.net = _DiffusionTSDenoiser(seq_len, n_features, d_model, n_heads, n_layers)

    def denoise(self, x_t: Tensor, t: Tensor) -> Tensor:
        return self.net(x_t, t)

    def loss(self, x0: Tensor) -> Tensor:
        return self.diffusion.ddpm_loss(self.denoise, x0)

    @torch.no_grad()
    def generate(self, n: int, device: str = "cpu") -> Tensor:
        shape = (n, self.seq_len, self.n_features)
        return self.diffusion.p_sample_loop(self.denoise, shape, device=device).cpu()
