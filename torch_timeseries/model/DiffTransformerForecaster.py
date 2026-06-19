"""DiffTransformerForecaster — Differential Transformer for time-series forecasting.

The Differential Transformer (Microsoft Research, 2024) cancels noise in attention
maps by computing the *difference* of two parallel softmax attention heads:

    DiffAttn(Q, K, V) = (softmax(Q₁·K₁ᵀ/√d') - λ·softmax(Q₂·K₂ᵀ/√d')) · V

where Q = [Q₁; Q₂], K = [K₁; K₂] (split along head dim), d' = d_head/2.

λ is a *learned scalar per head*:
    λ = exp(λ_q1·λ_k1) - exp(λ_q2·λ_k2) + λ_init       (λ_init ≈ 0.8)
which is re-parameterised via four learnable vectors.

The differential cancels "attention noise" — diffuse, low-signal attention weights
that spread uniformly across positions — leaving the genuine signal-discriminative
part of the attention maps intact.

Reference: Ye et al., "Differential Transformer", arXiv 2410.05258, 2024.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DiffAttnHead(nn.Module):
    """Single multi-head Differential Attention block."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, lambda_init: float = 0.8) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        assert (d_model // n_heads) % 2 == 0, "d_model//n_heads must be even (split into two sub-heads)"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads        # full head dim
        self.d_sub = self.d_head // 2           # sub-head dim for each of Q1, Q2, K1, K2
        self.scale = math.sqrt(self.d_sub)
        self.lambda_init = lambda_init

        # Q1/Q2 share input projection; same for K1/K2
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # Per-head λ re-parameterisation (4 scalars each)
        self.lambda_q1 = nn.Parameter(torch.randn(n_heads, self.d_sub) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(n_heads, self.d_sub) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(n_heads, self.d_sub) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(n_heads, self.d_sub) * 0.1)

        self.norm = nn.GroupNorm(n_heads, d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _lambda(self) -> torch.Tensor:
        # λ = exp(q1·k1) - exp(q2·k2) + lambda_init  → per head scalar (H,)
        lam = (
            torch.exp((self.lambda_q1 * self.lambda_k1).sum(-1))
            - torch.exp((self.lambda_q2 * self.lambda_k2).sum(-1))
            + self.lambda_init
        )
        return lam  # (H,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh, Ds = self.n_heads, self.d_head, self.d_sub

        # Q, K split into sub-heads; V is full head
        Q = self.W_Q(x).view(B, T, H, Dh)
        K = self.W_K(x).view(B, T, H, Dh)
        V = self.W_V(x).view(B, T, H, Dh)

        Q1, Q2 = Q[..., :Ds], Q[..., Ds:]   # (B,T,H,Ds) each
        K1, K2 = K[..., :Ds], K[..., Ds:]

        # Attention scores: (B, H, T, T)
        scores1 = torch.einsum("bthd,bshd->bhts", Q1, K1) / self.scale
        scores2 = torch.einsum("bthd,bshd->bhts", Q2, K2) / self.scale

        a1 = F.softmax(scores1, dim=-1)
        a2 = F.softmax(scores2, dim=-1)

        # Differential: (a1 - λ·a2) with per-head λ
        lam = self._lambda().view(1, H, 1, 1)      # broadcast
        diff = a1 - lam * a2                        # (B, H, T, T)
        diff = self.drop(diff)

        # Attend to values
        V_t = V.permute(0, 2, 1, 3)               # (B, H, T, Dh)
        out = torch.einsum("bhts,bhsd->bhtd", diff, V_t)   # (B, H, T, Dh)

        # GroupNorm over heads: reshape to (B, D, T) then back
        out = out.permute(0, 1, 3, 2).reshape(B, D, T)  # (B, D, T) for GroupNorm
        out = self.norm(out)
        out = out.permute(0, 2, 1)                 # (B, T, D)

        # Scale by (1 - lambda_init) as per paper
        out = out * (1.0 - self.lambda_init)
        out = self.drop(self.W_O(out))
        return self.out_norm(out + x)


class _DiffTransformerBlock(nn.Module):
    """DiffAttn block + GELU-FFN + residual."""

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float) -> None:
        super().__init__()
        self.attn = _DiffAttnHead(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        return x + self.ffn(x)


class DiffTransformerForecaster(nn.Module):
    """Differential Transformer multivariate forecaster.

    Channel-independent (CI) + RevIN.

    Args:
        seq_len:   input sequence length
        pred_len:  forecast horizon
        enc_in:    number of input channels
        d_model:   model dimension (must be divisible by 2·n_heads)
        n_heads:   number of attention heads
        d_ffn:     feedforward hidden size
        n_layers:  number of DiffTransformer blocks
        dropout:   dropout rate
        revin:     apply RevIN instance normalisation
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        n_heads: int = 4,
        d_ffn: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        revin: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert (d_model // n_heads) % 2 == 0, "d_model//n_heads must be even"
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.revin = revin

        if revin:
            from torch_timeseries.nn.revin import RevIN
            self.rev = RevIN(enc_in)

        self.embed = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            _DiffTransformerBlock(d_model, n_heads, d_ffn, dropout)
            for _ in range(n_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin:
            x = self.rev(x, "norm")

        B, T, C = x.shape
        x_ci = x.permute(0, 2, 1).reshape(B * C, T, 1)
        h = self.embed(x_ci)                    # (BC, T, d)

        for block in self.blocks:
            h = block(h)

        out = self.head(self.drop(h.mean(dim=1)))    # (BC, pred_len)
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)

        if self.revin:
            out = self.rev(out, "denorm")
        return out
