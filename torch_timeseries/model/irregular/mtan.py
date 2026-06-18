"""Multi-Time Attention Network (mTAN).

Reference: Shukla & Marlin, 2021 — "Multi-Time Attention Networks for
Irregularly Sampled Time Series".
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class _TimeEmbedding(nn.Module):
    """Sine/cosine time embedding with learned frequencies and phases."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.w = nn.Parameter(torch.randn(embed_dim // 2))
        self.b = nn.Parameter(torch.randn(embed_dim // 2))

    def forward(self, t: Tensor) -> Tensor:
        # t: (...,) → (..., embed_dim)
        t = t.unsqueeze(-1)                              # (..., 1)
        arg = t * self.w + self.b                        # (..., D/2)
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)


class _mTANEncoder(nn.Module):
    """Encodes irregular observations to reference time points via cross-attention."""

    def __init__(self, input_size: int, hidden_size: int,
                 num_ref_points: int, num_heads: int,
                 time_embed_dim: int) -> None:
        super().__init__()
        self.num_ref_points = num_ref_points
        self.hidden_size = hidden_size
        self.time_embed = _TimeEmbedding(time_embed_dim)

        self.ref_times = nn.Parameter(torch.linspace(0.0, 1.0, num_ref_points))

        self.in_proj = nn.Linear(input_size + time_embed_dim, hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.1, batch_first=True)

    def forward(self, x: Tensor, t: Tensor, mask: Tensor) -> Tensor:
        B, T, F = x.shape

        # Weight input by per-timestep observation rate
        t_emb = self.time_embed(t)                        # (B, T, D)
        obs_weight = mask.mean(dim=-1, keepdim=True)      # (B, T, 1)
        x_in = torch.cat([x * obs_weight, t_emb], dim=-1) # (B, T, F+D)
        v = self.in_proj(x_in)                             # (B, T, H)

        # Reference time embeddings as queries
        ref_emb = self.time_embed(
            self.ref_times.unsqueeze(0).expand(B, -1))    # (B, R, D)
        ref_in = torch.cat([
            torch.zeros(B, self.num_ref_points, F, device=x.device),
            ref_emb,
        ], dim=-1)                                         # (B, R, F+D)
        q = self.in_proj(ref_in)                           # (B, R, H)

        # True = "ignore this key". Guard against all-True rows (causes NaN in softmax).
        key_pad = (mask.sum(dim=-1) == 0)                  # (B, T) all-missing positions
        all_masked = key_pad.all(dim=-1, keepdim=True)     # (B, 1)
        key_pad = key_pad & ~all_masked                     # unblock all for fully-masked samples

        attn_out, _ = self.attn(q, v, v, key_padding_mask=key_pad)
        return attn_out                                    # (B, R, H)


class mTAN(nn.Module):
    """Multi-Time Attention Network for irregular time series.

    - ``forward(x, t, mask)`` → ``(B, output_size)`` for classification.
    - ``forward(x, t, mask, t_query=t_q)`` → ``(B, Tq, input_size)`` for
      interpolation and forecasting (seq2seq mode).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_ref_points: int = 16,
        num_heads: int = 2,
        time_embed_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = _mTANEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_ref_points=num_ref_points,
            num_heads=num_heads,
            time_embed_dim=time_embed_dim,
        )
        self.time_embed_dim = time_embed_dim
        self.drop = nn.Dropout(dropout)

        # Classification head: flatten reference outputs → output_size
        self.fc_cls = nn.Linear(num_ref_points * hidden_size, output_size)

        # Seq2Seq decoder: decode at arbitrary query times
        self.time_embed_dec = _TimeEmbedding(time_embed_dim)
        self.dec_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.fc_dec = nn.Linear(hidden_size + time_embed_dim, input_size)

    def forward(
        self,
        x: Tensor,              # (B, T, F)
        t: Tensor,              # (B, T)
        mask: Tensor,           # (B, T, F)
        x_time: Tensor = None,  # ignored
        t_query: Tensor = None, # (B, Tq) — enables seq2seq mode
    ) -> Tensor:
        ref = self.encoder(x, t, mask)    # (B, R, H)
        B, R, H = ref.shape

        if t_query is None:
            flat = ref.reshape(B, R * H)
            return self.fc_cls(self.drop(flat))   # (B, output_size)

        # Seq2Seq: attend from query time embeddings to reference representations
        Tq = t_query.shape[1]
        t_q_emb = self.time_embed_dec(t_query)     # (B, Tq, D)

        # Pad time embedding to H for cross-attention query dimension
        q_proj = x.new_zeros(B, Tq, H)
        min_d = min(H, t_q_emb.shape[-1])
        q_proj[:, :, :min_d] = t_q_emb[:, :, :min_d]

        attn_out, _ = self.dec_attn(q_proj, ref, ref)      # (B, Tq, H)
        combined = torch.cat([attn_out, t_q_emb], dim=-1)  # (B, Tq, H+D)
        return self.fc_dec(self.drop(combined))             # (B, Tq, F)
