"""SparseTransformerForecaster: Fixed local + strided sparse attention for time series.

Key idea (Longformer / BigBird / Child et al. 2019):
  Full O(T²) attention is expensive.  A sparse mask that combines:

    - **Local window**: token i attends to {i-w, …, i+w}   (context)
    - **Strided global tokens**: every *stride*-th token is a global token
      that attends to and is attended by all other tokens.

  gives O(T · (2w + T/stride)) complexity while preserving both fine-grained
  local dependencies and long-range context via the stride tokens.

Architecture comparison:
  VanillaTransformer:         full O(T²) attention.
  LinearAttentionForecaster:  O(Td²) via ELU+1 kernel — learned approximation.
  SparseTransformerForecaster: O(T·(w+T/s)) deterministic sparse mask — no learned kernel.

Architecture::

    Input (B, T, C)
      → RevIN norm
      → CI reshape → (B·C, T)
      → patch embedding  Linear(p, d_model) → (B·C, n_patches, d_model)
      → positional embedding (learnable)
      → L sparse-attention transformer layers
      → mean pool → (B·C, d_model)
      → Linear head → (B·C, pred_len)
      → reshape → (B, pred_len, C)
      → RevIN denorm

Sparse mask (built once in __init__, registered as a buffer):
  For n_patches tokens, position i attends to position j iff:
    |i - j| ≤ local_window  OR  (i % stride == 0) OR (j % stride == 0)

Args:
    seq_len:       input lookback T.
    pred_len:      forecast horizon.
    enc_in:        number of variates C.
    patch_size:    size of each patch (T must be divisible by patch_size after padding).
    d_model:       model dimension.
    n_heads:       number of attention heads.
    e_layers:      number of transformer layers.
    d_ff:          feed-forward hidden size.
    local_window:  half-width of local attention window.
    stride:        period of global/strided tokens.
    dropout:       dropout rate.
    revin:         use RevIN instance normalisation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.revin import RevIN


def _build_sparse_mask(n: int, local_window: int, stride: int, device=None) -> torch.Tensor:
    """Build additive attention bias mask: 0 = attend, -inf = mask out."""
    # Start with all masked
    mask = torch.full((n, n), float("-inf"), device=device)
    for i in range(n):
        # Local window
        lo = max(0, i - local_window)
        hi = min(n, i + local_window + 1)
        mask[i, lo:hi] = 0.0
        # Global (strided) tokens — token is global if its index is a stride multiple
        if i % stride == 0:
            mask[i, :] = 0.0   # global token attends to all
            mask[:, i] = 0.0   # all attend to global token
    return mask  # (n, n)


class _SparseSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_patches: int, local_window: int, stride: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        # Fixed sparse mask registered as buffer (shared across heads & batch)
        mask = _build_sparse_mask(n_patches, local_window, stride)
        self.register_buffer("sparse_mask", mask)  # (n, n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n, d_model) → (B, n, d_model)"""
        B, n, D = x.shape
        H, d = self.n_heads, self.d_head

        Q = self.q_proj(x).reshape(B, n, H, d).transpose(1, 2)  # (B,H,n,d)
        K = self.k_proj(x).reshape(B, n, H, d).transpose(1, 2)
        V = self.v_proj(x).reshape(B, n, H, d).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d)           # (B,H,n,n)
        scores = scores + self.sparse_mask.unsqueeze(0).unsqueeze(0)  # broadcast
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)                       # rows all-inf → 0
        attn = self.drop(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, n, D)
        return self.out_proj(out)


class _SparseTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_patches: int, d_ff: int,
                 local_window: int, stride: int, dropout: float):
        super().__init__()
        self.attn = _SparseSelfAttention(d_model, n_heads, n_patches, local_window, stride, dropout)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.drop(self.ff2(F.gelu(self.ff1(self.norm2(x)))))
        return x


class SparseTransformerForecaster(nn.Module):
    """Fixed local+strided sparse attention forecaster (channel-independent)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_size: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 128,
        local_window: int = 3,
        stride: int = 4,
        dropout: float = 0.1,
        revin: bool = True,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in
        self._revin_flag = revin

        if revin:
            self.revin_layer = RevIN(enc_in, affine=True)

        # Pad seq_len to be divisible by patch_size
        self.patch_size = patch_size
        pad = (patch_size - seq_len % patch_size) % patch_size
        self.pad = pad
        n_patches = (seq_len + pad) // patch_size

        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList([
            _SparseTransformerLayer(d_model, n_heads, n_patches, d_ff, local_window, stride, dropout)
            for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, pred_len, C)"""
        B, T, C = x.shape

        if self._revin_flag:
            x = self.revin_layer(x, "norm")

        # CI: (B*C, T)
        x_ci = x.permute(0, 2, 1).reshape(B * C, T)

        # Pad if needed
        if self.pad > 0:
            x_ci = F.pad(x_ci, (0, self.pad))

        # Patch embedding: (BC, T', 1) → (BC, n_patches, patch_size) → embed
        n_patches = x_ci.shape[-1] // self.patch_size
        x_p = x_ci.reshape(B * C, n_patches, self.patch_size)  # (BC, n, p)
        h = self.patch_embed(x_p) + self.pos_embed              # (BC, n, d_model)
        h = self.drop(h)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        # Mean pool over patches
        feat = h.mean(dim=1)                                     # (BC, d_model)
        pred = self.head(feat)                                   # (BC, pred_len)
        pred = pred.reshape(B, C, self.pred_len).permute(0, 2, 1)  # (B, pred_len, C)

        if self._revin_flag:
            pred = self.revin_layer(pred, "denorm")
        return pred
