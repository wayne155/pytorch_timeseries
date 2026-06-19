"""ConformerForecaster — Conformer (CNN + Transformer) hybrid forecaster.

Conformer block structure (Gulati et al. 2020):
    ½·FFN → MHSA → Conv → ½·FFN → LayerNorm

Convolution Module:
    LayerNorm → Pointwise(×2) → GLU → Depthwise-Conv → BN → Swish → Pointwise

The macaron-style half-weighted FFNs sandwich a multi-head attention and a
depthwise convolutional sub-block, coupling local and global temporal patterns
in each layer.

Reference: Gulati et al., "Conformer: Convolution-augmented Transformer for
           Speech Recognition" (2020)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _FeedForwardModule(nn.Module):
    """Macaron-style FFN: LN → Linear(d→4d) → Swish → Dropout → Linear(4d→d) → Dropout."""

    def __init__(self, d_model: int, d_ffn: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.fc2 = nn.Linear(d_ffn, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.silu(self.fc1(h))       # Swish = SiLU
        h = self.drop(h)
        h = self.drop(self.fc2(h))
        return h


class _ConvModule(nn.Module):
    """Conformer convolution sub-block: LN → PW-GLU → DW-Conv → BN → Swish → PW."""

    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        pad = (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(d_model)
        # Pointwise expansion → GLU halves to d_model
        self.pw_expand = nn.Conv1d(d_model, 2 * d_model, 1)
        # Depthwise conv (channel-wise temporal)
        self.dw_conv = nn.Conv1d(d_model, d_model, kernel_size, padding=pad,
                                 groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        # Pointwise projection back
        self.pw_proj = nn.Conv1d(d_model, d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        h = self.norm(x).transpose(1, 2)    # (B, d, T)
        h = self.pw_expand(h)               # (B, 2d, T)
        h = F.glu(h, dim=1)                 # (B, d, T) — GLU halves dim
        h = self.dw_conv(h)                 # (B, d, T)
        h = F.silu(self.bn(h))
        h = self.drop(self.pw_proj(h))
        return h.transpose(1, 2)            # (B, T, d)


class _ConformerBlock(nn.Module):
    """One Conformer block: ½FFN → MHSA → Conv → ½FFN → LN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ffn1 = _FeedForwardModule(d_model, d_ffn, dropout)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = _ConvModule(d_model, kernel_size, dropout)
        self.ffn2 = _FeedForwardModule(d_model, d_ffn, dropout)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Macaron ½FFN
        x = x + 0.5 * self.ffn1(x)
        # MHSA
        h = self.attn_norm(x)
        h, _ = self.attn(h, h, h)
        x = x + self.attn_drop(h)
        # Conv module
        x = x + self.conv(x)
        # Macaron ½FFN
        x = x + 0.5 * self.ffn2(x)
        return self.norm_out(x)


class ConformerForecaster(nn.Module):
    """Conformer multivariate forecaster.

    Channel-independent (CI) + RevIN.

    Args:
        seq_len:     input sequence length
        pred_len:    forecast horizon
        enc_in:      number of input channels
        d_model:     model dimension (must be divisible by n_heads)
        n_heads:     number of attention heads
        d_ffn:       feedforward hidden dimension
        n_layers:    number of Conformer blocks
        kernel_size: depthwise-conv kernel size (odd integer)
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
        d_ffn: int = 256,
        n_layers: int = 2,
        kernel_size: int = 9,
        dropout: float = 0.1,
        revin: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.revin = revin

        if revin:
            from torch_timeseries.nn.revin import RevIN
            self.rev = RevIN(enc_in)

        self.embed = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            _ConformerBlock(d_model, n_heads, d_ffn, kernel_size, dropout)
            for _ in range(n_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        if self.revin:
            x = self.rev(x, "norm")

        B, T, C = x.shape
        x_ci = x.permute(0, 2, 1).reshape(B * C, T, 1)
        h = self.embed(x_ci)                # (BC, T, d)

        for block in self.blocks:
            h = block(h)

        out = self.head(self.drop(h.mean(dim=1)))     # (BC, pred_len)
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)

        if self.revin:
            out = self.rev(out, "denorm")
        return out
