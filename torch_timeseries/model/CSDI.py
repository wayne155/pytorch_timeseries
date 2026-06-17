"""CSDI: Conditional Score-based Diffusion Imputation (Tashiro et al., 2021).

For unconditional generation, cond=None (zero conditioning).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .diffusion_utils import GaussianDiffusion, sinusoidal_embedding


class _DualAttentionBlock(nn.Module):
    """One residual block with interleaved temporal and feature attention."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.t_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.f_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.ff     = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)
        )
        self.norm3  = nn.LayerNorm(d_model)

    def forward(self, h: Tensor) -> Tensor:
        # h: (B, T, C, D)
        B, T, C, D = h.shape
        # temporal attention over T, per feature channel
        ht = h.permute(0, 2, 1, 3).reshape(B * C, T, D)
        ht = self.norm1(ht + self.t_attn(ht, ht, ht, need_weights=False)[0])
        h = ht.reshape(B, C, T, D).permute(0, 2, 1, 3)
        # feature attention over C, per time step
        hf = h.reshape(B * T, C, D)
        hf = self.norm2(hf + self.f_attn(hf, hf, hf, need_weights=False)[0])
        h = hf.reshape(B, T, C, D)
        # feed-forward
        h = self.norm3(h + self.ff(h))
        return h


class _CSDIDenoiser(nn.Module):
    """Score network epsilon_theta(x_t, t, cond)."""

    def __init__(self, n_features: int, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.in_proj  = nn.Linear(2, d_model)
        self.t_proj   = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.blocks   = nn.ModuleList(
            [_DualAttentionBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x_t: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        # x_t, cond: (B, T, C)
        h = self.in_proj(torch.stack([x_t, cond], dim=-1))   # (B, T, C, D)
        D = h.shape[-1]
        t_emb = self.t_proj(sinusoidal_embedding(t, D))       # (B, D)
        h = h + t_emb[:, None, None, :]
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h).squeeze(-1)                   # (B, T, C)


class CSDI(nn.Module):
    """CSDI — Conditional Score-based Diffusion for time series (Tashiro et al., NeurIPS 2021).

    Score-based DDPM denoiser with a dual-attention Transformer that attends
    independently over the time axis and the feature axis.  Originally designed
    for probabilistic imputation (conditioning on observed values), but also
    used here for unconditional generation by setting ``cond=None``.

    Paper: *CSDI: Conditional Score-based Diffusion Models for Probabilistic
    Time Series Imputation.*
    https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html

    Args:
        seq_len (int): Sequence length of each window.
        n_features (int): Number of channels.
        d_model (int): Transformer hidden dimension. Defaults to 64.
        n_heads (int): Number of attention heads. Defaults to 8.
        n_layers (int): Number of dual-attention blocks. Defaults to 4.
        T (int): Number of diffusion steps. Defaults to 100.
        schedule (str): Beta schedule — ``'linear'`` or ``'cosine'``.
            Defaults to ``'linear'``.

    Tasks: Generation.
    """

    def __init__(self, seq_len: int, n_features: int,
                 d_model: int = 64, n_heads: int = 8, n_layers: int = 4,
                 T: int = 100, schedule: str = "linear"):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.diffusion = GaussianDiffusion(T=T, schedule=schedule)
        self.score_net = _CSDIDenoiser(n_features, d_model, n_heads, n_layers)

    def denoise(self, x_t: Tensor, t: Tensor, cond: Tensor = None) -> Tensor:
        if cond is None:
            cond = torch.zeros_like(x_t)
        return self.score_net(x_t, t, cond)

    def loss(self, x0: Tensor) -> Tensor:
        return self.diffusion.ddpm_loss(self.denoise, x0)

    @torch.no_grad()
    def generate(self, n: int, device: str = "cpu") -> Tensor:
        shape = (n, self.seq_len, self.n_features)
        return self.diffusion.p_sample_loop(self.denoise, shape, device=device).cpu()
