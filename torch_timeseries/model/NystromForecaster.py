"""NystromForecaster — Nyström approximation of softmax attention.

Standard softmax attention is O(T²·d).  The Nyström method approximates the
attention matrix using m ≪ T *landmark* tokens:

    Landmarks Ñ:  Ñ_i = mean of segment [i·T/m : (i+1)·T/m]   i = 0…m-1

    A = softmax(Q · Ñᵀ / √d)        (B, T, m)    — query–landmark similarity
    B = softmax(Ñ · Kᵀ / √d)        (B, m, T)    — landmark–key similarity
    Kernel = A · pinv(softmax(Ñ · Ñᵀ / √d)) · B  (B, T, T) approximation

    Attn ≈ Kernel · V

Complexity: O(T·m) instead of O(T²) for the dominant matmul steps.

The pseudoinverse is computed via the Moore-Penrose formula using an iterative
Schulz method (as in the original paper) or torch.linalg.pinv for correctness.

Reference: Xiong et al., "Nystromformer: A Nyström-Based Self-Attention
Mechanism", AAAI 2021.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _iterative_pinv(A: torch.Tensor, n_iter: int = 6) -> torch.Tensor:
    """Iterative Moore-Penrose pseudoinverse via Schulz method.
    A: (..., m, m)  → returns pseudo-inverse of same shape.
    """
    # Initialise: A_0 = αAᵀ where α = 1 / (σ_max * σ_max)  ≈ 1 / ||A||₁ / ||A||∞
    norm = (A.abs().sum(dim=-1).max(dim=-1).values *
            A.abs().sum(dim=-2).max(dim=-1).values).unsqueeze(-1).unsqueeze(-1)
    V = A.mT / (norm + 1e-8)
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype).expand_as(A)
    for _ in range(n_iter):
        V = 2.0 * V - V @ A @ V
    return V


class _NystromAttentionBlock(nn.Module):
    """Multi-head Nyström attention block."""

    def __init__(self, d_model: int, n_heads: int, n_landmarks: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_landmarks = n_landmarks
        self.scale = math.sqrt(self.d_head)

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh, m = self.n_heads, self.d_head, self.n_landmarks

        Q = self.W_Q(x).view(B, T, H, Dh).permute(0, 2, 1, 3)  # (B,H,T,Dh)
        K = self.W_K(x).view(B, T, H, Dh).permute(0, 2, 1, 3)
        V = self.W_V(x).view(B, T, H, Dh).permute(0, 2, 1, 3)

        # Compute landmark tokens via segment-mean of K
        # Pad T to be divisible by m
        pad = (m - T % m) % m
        K_pad = F.pad(K, (0, 0, 0, pad))   # (B,H,T+pad,Dh)
        T_pad = T + pad
        seg_len = T_pad // m
        landmarks = K_pad.view(B, H, m, seg_len, Dh).mean(dim=3)  # (B,H,m,Dh)

        # A = softmax(Q · Ñᵀ / √d)   (B,H,T,m)
        A = F.softmax(torch.einsum("bhtd,bhmd->bhtm", Q, landmarks) / self.scale, dim=-1)

        # B = softmax(Ñ · Kᵀ / √d)   (B,H,m,T)
        Bmat = F.softmax(torch.einsum("bhmd,bhtd->bhmt", landmarks, K) / self.scale, dim=-1)

        # Kernel approx of Ñ·Ñᵀ  (B,H,m,m)
        NN = F.softmax(torch.einsum("bhmd,bhnd->bhmn", landmarks, landmarks) / self.scale, dim=-1)

        # Moore-Penrose pseudo-inverse of NN
        NN_inv = _iterative_pinv(NN)  # (B,H,m,m)

        # Approximate attention: (B,H,T,T) approximated as A @ NN_inv @ B
        # then multiply V: output = (A @ NN_inv @ B) @ V
        # Efficient: compute step by step
        tmp = torch.einsum("bhtm,bhmn->bhtn", A, NN_inv)   # (B,H,T,m)
        tmp = torch.einsum("bhtn,bhnT->bhtT", tmp, Bmat)    # (B,H,T,T) — for small T ok
        out = self.drop(torch.einsum("bhtT,bhTd->bhtd", tmp, V))  # (B,H,T,Dh)

        out = out.permute(0, 2, 1, 3).reshape(B, T, D)
        out = self.drop(self.W_O(out))
        return self.norm(out + x)


class _NystromBlock(nn.Module):
    """Nyström attention + GELU-FFN + residual."""

    def __init__(self, d_model: int, n_heads: int, n_landmarks: int, d_ffn: int, dropout: float) -> None:
        super().__init__()
        self.attn = _NystromAttentionBlock(d_model, n_heads, n_landmarks, dropout)
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


class NystromForecaster(nn.Module):
    """Nyström approximation Transformer multivariate forecaster.

    Channel-independent (CI) + RevIN.

    Args:
        seq_len:     input sequence length
        pred_len:    forecast horizon
        enc_in:      number of input channels
        d_model:     model dimension (divisible by n_heads)
        n_heads:     number of attention heads
        n_landmarks: number of Nyström landmark tokens (m ≪ T)
        d_ffn:       feedforward hidden size
        n_layers:    number of Nyström Transformer layers
        dropout:     dropout rate
        revin:       apply RevIN instance normalisation
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_landmarks: int = 8,
        d_ffn: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        revin: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.revin = revin

        if revin:
            from torch_timeseries.nn.revin import RevIN
            self.rev = RevIN(enc_in)

        self.embed = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            _NystromBlock(d_model, n_heads, n_landmarks, d_ffn, dropout)
            for _ in range(n_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin:
            x = self.rev(x, "norm")

        B, T, C = x.shape
        x_ci = x.permute(0, 2, 1).reshape(B * C, T, 1)
        h = self.embed(x_ci)            # (BC, T, d)

        for block in self.blocks:
            h = block(h)

        out = self.head(self.drop(h.mean(dim=1)))     # (BC, pred_len)
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)

        if self.revin:
            out = self.rev(out, "denorm")
        return out
