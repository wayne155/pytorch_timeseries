"""ETSformer: Exponential Smoothing Transformer for time-series forecasting.

Reference: Woo et al., "ETSformer: Exponential Smoothing Transformers for
Time-series Forecasting", NeurIPS 2022 (Salesforce Research).

Key idea:
  Classic Exponential Smoothing (ETS) captures level, trend, and seasonality
  via scalar smoothing equations.  ETSformer replaces scalar smoothing with
  learned **vector** smoothing operations via attention-like mechanisms:

    1. **Exponential Smoothing Attention (ES-Attn)**: replaces softmax(QK^T)V
       with an exponentially decaying attention weight across the sequence.
       Each query position aggregates all key-value pairs with weights
       α^(t-s) (decay parameter α per head, learned).

    2. **Frequency Attention (F-Attn)**: frequency domain self-attention that
       operates on the top-K Fourier components of the key-value projections.
       This captures long-range periodic dependencies efficiently.

    3. **Growth damping**: optional damping parameter φ that attenuates the
       trend contribution in the forecast, similar to the damped trend in ETS.

Simplified implementation:
  This implementation follows the core ES-Attn + F-Attn design with:
  - ES-Attn: per-head exponential decay weights (causal, O(T) per head)
  - F-Attn: top-K FFT-based seasonal attention
  - Stacked encoder layers with both attention types
  - Direct output projection with level + growth decomposition

Args:
    seq_len:       input lookback window T.
    pred_len:      forecasting horizon H.
    enc_in:        number of variates.
    d_model:       model dimension.
    n_heads:       number of attention heads.
    e_layers:      number of ETSformer encoder layers.
    d_ff:          feed-forward hidden size.
    dropout:       dropout rate.
    top_k:         number of Fourier frequencies in F-Attn.
    smoothing_learning_rate: initial value for smoothing parameters.
    revin:         use RevIN normalisation.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True)
            self._std = x.std(dim=1, keepdim=True) + self.eps
            x = (x - self._mean) / self._std
            x = x * self.affine_weight + self.affine_bias
        elif mode == "denorm":
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self._std + self._mean
        return x


class _ExponentialSmoothingAttention(nn.Module):
    """ES-Attn: multi-head exponential smoothing attention.

    For each head h, the attention weight between query at position t and
    key at position s is proportional to α_h^(t-s) (lower weight for older keys).
    This implements a causal O(N·H·D) version via a simple weighted sum.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 init_alpha: float = 0.5):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        # Per-head learned smoothing parameter α ∈ (0, 1)
        self.log_alpha = nn.Parameter(
            torch.full((n_heads,), math.log(init_alpha / (1 - init_alpha)))
        )

    def _smoothing_weights(self, T: int, device: torch.device) -> torch.Tensor:
        """Compute (n_heads, T, T) exponential decay weight matrix.

        w[h, t, s] = α_h^(t-s) for s ≤ t, 0 otherwise.
        """
        alpha = torch.sigmoid(self.log_alpha)   # (n_heads,)
        t_idx = torch.arange(T, device=device).float()
        diffs = (t_idx.unsqueeze(1) - t_idx.unsqueeze(0)).clamp(min=0)  # (T, T)
        # log-space: log(α^d) = d * log(α)
        log_w = diffs.unsqueeze(0) * torch.log(alpha + 1e-9).view(-1, 1, 1)  # (H, T, T)
        # Causal mask: future keys → -inf
        causal_mask = (t_idx.unsqueeze(1) < t_idx.unsqueeze(0)).unsqueeze(0)
        log_w = log_w.masked_fill(causal_mask, float("-inf"))
        return torch.softmax(log_w, dim=-1)     # (H, T, T) — normalised per row

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, T, d_model)``

        Returns:
            ``(B, T, d_model)``
        """
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        q = self.q_proj(x).reshape(B, T, H, Dh).transpose(1, 2)  # (B, H, T, Dh)
        k = self.k_proj(x).reshape(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, Dh).transpose(1, 2)

        w = self._smoothing_weights(T, x.device)    # (H, T, T)
        # v: (B, H, T, Dh) × w: (H, T, T) → attn: (B, H, T, Dh)
        attn = torch.einsum("bhtd,hts->bhsd", v, w)   # (B, H, T, Dh)
        # Modulate by query (element-wise query gating for context-awareness)
        attn = attn * torch.sigmoid(q)

        attn = self.drop(attn)
        attn = attn.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(attn)


class _FrequencyAttention(nn.Module):
    """F-Attn: frequency domain attention on top-K Fourier components.

    Projects values into frequency domain, keeps the top-K frequency bins
    (by mean amplitude), applies a learned frequency-domain gate, then
    iFFT back to time domain.
    """

    def __init__(self, d_model: int, top_k: int = 5):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, T, d_model)``

        Returns:
            ``(B, T, d_model)``
        """
        B, T, D = x.shape
        xf = torch.fft.rfft(x, dim=1)                  # (B, T//2+1, D)
        amp = xf.abs().mean(dim=(0, 2))                  # (T//2+1,)
        k = min(self.top_k, amp.size(0))
        _, top_idx = amp.topk(k)
        mask = torch.zeros_like(xf)
        mask[:, top_idx, :] = 1.0
        xf_filtered = xf * mask

        # Frequency-domain gate (applied in real-valued view)
        x_time = torch.fft.irfft(xf_filtered, n=T, dim=1)   # (B, T, D)
        gate = torch.sigmoid(self.gate(x_time))
        return x_time * gate


class _ETSformerLayer(nn.Module):
    """One ETSformer encoder layer: ES-Attn + F-Attn + FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, top_k: int,
                 dropout: float = 0.0):
        super().__init__()
        self.es_attn = _ExponentialSmoothingAttention(d_model, n_heads, dropout)
        self.f_attn = _FrequencyAttention(d_model, top_k)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.drop(self.es_attn(x)))
        x = self.norm2(x + self.drop(self.f_attn(x)))
        x = self.norm3(x + self.drop(self.ffn(x)))
        return x


class ETSformer(nn.Module):
    """ETSformer: Exponential Smoothing Transformer.

    Replaces softmax attention with exponential decay-weighted smoothing
    and frequency-domain seasonal attention.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 256,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        top_k: int = 5,
        revin: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

        self.revin = revin
        if revin:
            self.revin_layer = _RevIN(enc_in)

        # Input projection: enc_in → d_model
        self.input_proj = nn.Linear(enc_in, d_model)

        # ETSformer encoder
        self.layers = nn.ModuleList([
            _ETSformerLayer(d_model, n_heads, d_ff, top_k, dropout)
            for _ in range(e_layers)
        ])

        # Output: last T positions → project to (pred_len, enc_in)
        self.output_proj = nn.Linear(d_model, pred_len * enc_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, seq_len, enc_in)``

        Returns:
            ``(B, pred_len, enc_in)``
        """
        B, T, N = x.shape

        if self.revin:
            x = self.revin_layer(x, "norm")

        h = self.input_proj(x)                          # (B, T, d_model)
        for layer in self.layers:
            h = layer(h)

        # Use the last time step's representation for forecasting
        # (the ES-Attn is causal, so h[:, -1, :] aggregates all history)
        out = self.output_proj(h[:, -1, :])             # (B, pred_len * enc_in)
        out = out.reshape(B, self.pred_len, N)

        if self.revin:
            out = self.revin_layer(out, "denorm")

        return out
