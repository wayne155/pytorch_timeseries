"""ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis.

Reference: Luo & Wang, "ModernTCN: A Modern Pure Convolution Structure for
General Time Series Analysis", ICLR 2024.

Key ideas:
  Classical TCN uses small kernels (3-5) in a causal, dilated stack.  ModernTCN
  re-examines this design with modern deep learning principles:

    1. **Patch Embedding (Stem)**: a strided convolution projects each
       per-variate 1-D input into d_model token embeddings, down-sampling
       the time axis by `patch_stride`.  Equivalent to a ViT/PatchTST stem
       but implemented as Conv1d.

    2. **Large Depthwise Convolution (DW-LargeConv)**: the main temporal
       mixing layer uses a very large kernel (kernel_size ≥ 51) applied
       depthwise (one filter per channel).  A single large-kernel conv can
       capture the same receptive field as many stacked small-kernel layers,
       but with fewer parameters and without causal masking.

    3. **ConvFFN**: a 2-layer pointwise conv block (1×1 conv → GELU → 1×1 conv)
       with 4× expansion, analogous to the Transformer FFN.  Mixes channel
       information after the depthwise conv.

    4. **BatchNorm + residuals**: each sub-block uses BatchNorm for training
       stability and skip connections for gradient flow.

  Overall pipeline (channel-independent, one SSM per variate):
    x → RevIN → [per-variate] Stem → ModernTCN blocks → AvgPool → Output proj
    → RevIN-denorm → forecast

Args:
    seq_len:      input lookback window.
    pred_len:     forecast horizon.
    enc_in:       number of variates.
    patch_size:   stem convolution kernel size.
    patch_stride: stride for the stem conv (down-sampling factor).
    d_model:      token embedding dimension.
    kernel_size:  large depthwise conv kernel (must be odd).
    e_layers:     number of ModernTCN blocks.
    d_ff:         ConvFFN expansion multiplier.
    dropout:      dropout rate.
    revin:        use RevIN instance normalisation.
"""
from __future__ import annotations

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
            self._mean = x.mean(1, keepdim=True)
            self._std = x.std(1, keepdim=True).clamp(self.eps)
            x = (x - self._mean) / self._std
            return x * self.affine_weight + self.affine_bias
        else:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            return x * self._std + self._mean


class _ModernTCNBlock(nn.Module):
    """One ModernTCN block: large-kernel DW-Conv + ConvFFN, each with BN + residual."""

    def __init__(self, d_model: int, kernel_size: int, d_ff: int, dropout: float):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd (same-length padding)"
        pad = (kernel_size - 1) // 2
        # Large depthwise conv (per-channel temporal mixing)
        self.dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, padding=pad, groups=d_model,
        )
        self.norm1 = nn.BatchNorm1d(d_model)
        # ConvFFN: pointwise expand → GELU → contract
        self.pw1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.pw2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B_C, d_model, T_patch)
        # DW-LargeConv sub-block
        res = x
        x = self.norm1(self.dw_conv(x))
        x = res + F.dropout(x, p=self.dropout, training=self.training)
        # ConvFFN sub-block
        res = x
        x = F.gelu(self.pw1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.norm2(self.pw2(x))
        return res + F.dropout(x, p=self.dropout, training=self.training)


class ModernTCN(nn.Module):
    """ModernTCN: large-kernel depthwise conv + patch stem (channel-independent)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_size: int = 8,
        patch_stride: int = 4,
        d_model: int = 128,
        kernel_size: int = 51,
        e_layers: int = 3,
        d_ff_ratio: int = 4,
        dropout: float = 0.05,
        revin: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self._revin_flag = revin
        if revin:
            self.revin_layer = _RevIN(enc_in)

        d_ff = d_model * d_ff_ratio

        # Stem: (B*C, 1, T) → (B*C, d_model, T_patch) via strided conv
        # Use padding to align output length
        pad_stem = (patch_size - 1) // 2
        self.stem = nn.Conv1d(1, d_model, kernel_size=patch_size,
                              stride=patch_stride, padding=pad_stem)
        self.stem_norm = nn.BatchNorm1d(d_model)

        self.blocks = nn.ModuleList(
            [_ModernTCNBlock(d_model, kernel_size, d_ff, dropout) for _ in range(e_layers)]
        )

        # Head: global avg-pool → linear to pred_len (channel-independent)
        self.output_proj = nn.Linear(d_model, pred_len)
        self.dropout = dropout

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

        # Channel-independent: (B, T, C) → (B, C, T) → (B*C, 1, T)
        x_ci = x.transpose(1, 2).reshape(B * C, 1, T)

        # Stem
        h = F.gelu(self.stem_norm(self.stem(x_ci)))  # (B*C, d_model, T_patch)

        for block in self.blocks:
            h = block(h)

        # Global average pool: (B*C, d_model, T_patch) → (B*C, d_model)
        h = h.mean(dim=-1)

        # Output projection: (B*C, d_model) → (B*C, pred_len)
        out = self.output_proj(h)                     # (B*C, pred_len)

        # Reshape: (B*C, pred_len) → (B, C, pred_len) → (B, pred_len, C)
        out = out.reshape(B, C, self.pred_len).transpose(1, 2)

        if self._revin_flag:
            out = self.revin_layer(out, "denorm")
        return out
