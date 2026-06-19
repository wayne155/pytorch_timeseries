"""CARD: Channel Aligned Robust Blend Transformer for time-series forecasting.

Reference: Wang et al., "CARD: Channel Aligned Robust Blend Transformer for
Time Series Forecasting", ICLR 2024.

Key ideas:
  Time-series channels often carry both *shared* global trends (best captured
  by cross-channel attention) and *private* local temporal patterns (best
  captured by channel-independent patch attention).  Existing models commit to
  one view.  CARD uses both simultaneously and blends them adaptively.

  1. **Temporal Encoder (per-channel patches)**:
     Each variate is independently cut into overlapping patches and processed
     by a standard Transformer encoder.  This mirrors PatchTST and is
     efficient because the sequence length fed to attention is n_patches, not T.

  2. **Channel Dependency Encoder**:
     Inspired by iTransformer: the entire time series of each variate is
     projected to a single d_model token.  Attention is then applied *across
     channels*, letting each variate attend to all others.  This captures
     multivariate dependencies without the quadratic-in-T cost.

  3. **Robust Blend Gate**:
     A per-position sigmoid gate g ∈ (0,1) is predicted from the concatenation
     of both encoder outputs and combines them:
         out = g * temporal_repr + (1-g) * channel_repr
     This lets the model lean on whichever encoder is more informative for each
     individual token / position.

  4. **RevIN** normalisation with learnable affine params.

Pipeline:
    x → RevIN → ┬─ Patch → TemporalEncoder  ─┐
                │                              ├→ BlendGate → OutputProj → denorm
                └─ ChanProj → ChannelEncoder ─┘

Args:
    seq_len:       input lookback window length.
    pred_len:      forecast horizon.
    enc_in:        number of input variates.
    d_model:       hidden dimension.
    n_heads:       attention heads (both encoders share the same setting).
    e_layers:      number of encoder layers (both encoders).
    d_ff:          feed-forward hidden size.
    patch_len:     patch length for the temporal encoder.
    stride:        stride between patches (default patch_len//2).
    dropout:       dropout rate.
    revin:         use RevIN instance normalisation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────


class _RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._mean = x.mean(1, keepdim=True)
            self._std = x.std(1, keepdim=True).clamp(self.eps)
            x = (x - self._mean) / self._std
            return x * self.affine_weight + self.affine_bias
        else:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            return x * self._std + self._mean


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.reshape(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.reshape(B, L, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = self.drop(torch.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class _FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.attn = _MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = _FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# CARD
# ──────────────────────────────────────────────────────────────────────────────


class CARD(nn.Module):
    """Channel Aligned Robust Blend Transformer."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        patch_len: int = 16,
        stride: int | None = None,
        dropout: float = 0.1,
        revin: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self._revin_flag = revin

        if revin:
            self.revin_layer = _RevIN(enc_in)

        if stride is None:
            stride = patch_len // 2
        self.patch_len = patch_len
        self.stride = stride

        # ── Temporal Encoder (per-channel patches) ───────────────────────────
        # Number of patches: ceiling of (seq_len - patch_len) / stride + 1
        # with end-padding so sequence covers seq_len exactly
        n_patches = math.ceil((seq_len - patch_len) / stride) + 1
        self.n_patches = n_patches

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.temporal_pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        self.temporal_layers = nn.ModuleList(
            [_TransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )
        self.temporal_norm = nn.LayerNorm(d_model)
        # Project patches → pred_len (per patch → pred_len contribution, then avg)
        # We'll do n_patches tokens → mean-pool → linear
        self.temporal_proj = nn.Linear(d_model, pred_len)

        # ── Channel Dependency Encoder ────────────────────────────────────────
        # Each channel: project seq_len → d_model (one token per variate)
        self.chan_proj = nn.Linear(seq_len, d_model)
        self.chan_layers = nn.ModuleList(
            [_TransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )
        self.chan_norm = nn.LayerNorm(d_model)
        self.chan_decoder = nn.Linear(d_model, pred_len)

        # ── Blend Gate ────────────────────────────────────────────────────────
        # Gate input: concatenation of temporal and channel representations
        # (B, C, 2*d_model) → gate (B, C, 1) per variate
        self.gate_proj = nn.Linear(2 * d_model, 1)

        # ── Dropout ───────────────────────────────────────────────────────────
        self.drop = nn.Dropout(dropout)

    def _patchify(self, x_ci: torch.Tensor) -> torch.Tensor:
        """Extract overlapping patches from channel-independent input.

        Args:
            x_ci: (B*C, T)
        Returns:
            (B*C, n_patches, patch_len)
        """
        BC, T = x_ci.shape
        # Pad the end so we have exactly n_patches patches
        pad_len = (self.n_patches - 1) * self.stride + self.patch_len - T
        if pad_len > 0:
            x_ci = F.pad(x_ci, (0, pad_len), mode="constant", value=0)
        patches = x_ci.unfold(-1, self.patch_len, self.stride)  # (BC, n_patches, patch_len)
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)
        Returns:
            (B, pred_len, C)
        """
        B, T, C = x.shape

        if self._revin_flag:
            x = self.revin_layer(x, "norm")

        # ── Temporal Encoder ──────────────────────────────────────────────────
        # channel-independent: (B, T, C) → (B*C, T) → patches
        x_ci = x.transpose(1, 2).reshape(B * C, T)
        patches = self._patchify(x_ci)                            # (B*C, n_patches, patch_len)
        h_t = self.patch_embed(patches) + self.temporal_pos       # (B*C, n_patches, d_model)
        h_t = self.drop(h_t)
        for layer in self.temporal_layers:
            h_t = layer(h_t)
        h_t = self.temporal_norm(h_t)                             # (B*C, n_patches, d_model)

        # Mean-pool patches → (B*C, d_model) → (B, C, d_model)
        h_t_pooled = h_t.mean(dim=1).reshape(B, C, -1)           # (B, C, d_model)
        temp_out = self.temporal_proj(h_t_pooled)                 # (B, C, pred_len)

        # ── Channel Dependency Encoder ────────────────────────────────────────
        # (B, T, C) → (B, C, T) → project each channel to d_model token
        x_chan = x.transpose(1, 2)                                # (B, C, T)
        h_c = self.chan_proj(x_chan)                              # (B, C, d_model)
        h_c = self.drop(h_c)
        for layer in self.chan_layers:
            h_c = layer(h_c)                                      # cross-channel attn
        h_c = self.chan_norm(h_c)                                 # (B, C, d_model)
        chan_out = self.chan_decoder(h_c)                         # (B, C, pred_len)

        # ── Blend Gate ────────────────────────────────────────────────────────
        # Predict gate from pooled temporal and channel representations
        gate_input = torch.cat([h_t_pooled, h_c], dim=-1)        # (B, C, 2*d_model)
        g = torch.sigmoid(self.gate_proj(gate_input))             # (B, C, 1)

        out = g * temp_out + (1 - g) * chan_out                  # (B, C, pred_len)
        out = out.transpose(1, 2)                                 # (B, pred_len, C)

        if self._revin_flag:
            out = self.revin_layer(out, "denorm")
        return out
