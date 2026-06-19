"""FastFormerForecaster — Fastformer additive attention forecaster.

Fastformer replaces O(T²) dot-product attention with an O(T·d) additive scheme:

1. Global query: α_t = softmax(w_a · q_t); q_g = Σ_t α_t ⊙ q_t
2. Key interaction: p_t = q_g ⊙ k_t            # element-wise × broadcasted global
3. Global key:   β_t = softmax(w_b · p_t); p_g = Σ_t β_t ⊙ p_t
4. Output:       u_t = W_o( p_g ⊙ v_t ) + x   # global context × local value

Each of the three aggregations is O(T·d), making the full block O(T·d) unlike
standard attention O(T²·d).

Reference: Wu et al., "Fastformer: Additive Attention Can Be All You Need" (2021)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _FastAttentionBlock(nn.Module):
    """Single Fastformer attention block."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        # Scalar attention weights for global pooling (one per head)
        self.w_a = nn.Linear(self.d_head, 1, bias=False)  # for global query
        self.w_b = nn.Linear(self.d_head, 1, bias=False)  # for global key
        self.W_O = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        Q = self.W_Q(x).view(B, T, H, Dh)     # (B, T, H, Dh)
        K = self.W_K(x).view(B, T, H, Dh)     # (B, T, H, Dh)
        V = self.W_V(x).view(B, T, H, Dh)     # (B, T, H, Dh)

        # Step 1: global query aggregation
        alpha = F.softmax(self.w_a(Q), dim=1)  # (B, T, H, 1)
        q_g = (alpha * Q).sum(dim=1, keepdim=True)  # (B, 1, H, Dh)

        # Step 2: pointwise interaction
        P = q_g * K                            # (B, T, H, Dh)

        # Step 3: global key aggregation
        beta = F.softmax(self.w_b(P), dim=1)   # (B, T, H, 1)
        p_g = (beta * P).sum(dim=1, keepdim=True)  # (B, 1, H, Dh)

        # Step 4: output = global context × value
        U = (p_g * V).view(B, T, D)            # (B, T, d)
        out = self.drop(self.W_O(U))

        return self.norm(out + x)


class _FastFormerLayer(nn.Module):
    """FastAttentionBlock + FFN + residual."""

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float) -> None:
        super().__init__()
        self.attn = _FastAttentionBlock(d_model, n_heads, dropout)
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


class FastFormerForecaster(nn.Module):
    """Fastformer (additive attention) multivariate forecaster.

    Channel-independent (CI) + RevIN.

    Args:
        seq_len:   input sequence length
        pred_len:  forecast horizon
        enc_in:    number of input channels
        d_model:   model dimension (must be divisible by n_heads)
        n_heads:   number of attention heads
        d_ffn:     feedforward dimension
        n_layers:  number of Fastformer layers
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
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.revin = revin

        if revin:
            from torch_timeseries.nn.revin import RevIN
            self.rev = RevIN(enc_in)

        self.embed = nn.Linear(1, d_model)
        self.layers = nn.ModuleList([
            _FastFormerLayer(d_model, n_heads, d_ffn, dropout)
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

        for layer in self.layers:
            h = layer(h)

        out = self.head(self.drop(h.mean(dim=1)))     # (BC, pred_len)
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)

        if self.revin:
            out = self.rev(out, "denorm")
        return out
