"""DishTS: Dish-TS for non-stationary time-series forecasting.

Reference: Fan et al., "Dish-TS: A General Paradigm for Temporal Distribution
Shifts in Time Series Forecasting", AAAI 2024.

Key idea:
  Non-stationary time series suffer from temporal distribution shift: the
  statistical properties (mean, variance) of the input window differ from
  those of the forecast horizon.  Simple instance normalisation (RevIN) uses
  the same statistics for both normalisation and de-normalisation, which is
  suboptimal when the shift is input-dependent.

  Dish-TS introduces the **Dish** (Distribution Shift) normalisation layer:

    1. **CoSta (Coefficient Statistics)**: learn two small networks that
       predict time-varying coefficient vectors (phi_B, phi_W) from the
       input series.  These play the roles of "bias" and "weight" corrections:
         phi_B predicts the *mean shift*  (replaces the simple mean subtraction)
         phi_W predicts the *scale shift* (replaces the simple std division)

    2. **Normalisation**: apply the learned shifts to the input:
         x_norm = (x - phi_B) / (phi_W + eps)

    3. **Backbone**: a standard Transformer encoder (any backbone works;
       here we use a lightweight version similar to PatchTST without patches).

    4. **De-normalisation**: apply the inverse transformation to the output
       using the *predicted future* shift coefficients (key novelty of Dish-TS:
       the CoSta network predicts phi for BOTH input and forecast windows).

Simplified implementation:
  - CoSta networks: 2-layer MLP (seq_len → hidden → 2*enc_in) per window.
    Outputs phi_B (mean proxy) and phi_W (scale proxy) per channel.
  - Backbone: 2-layer encoder-only Transformer with channel mixing.
  - Output: direct token projection from last position to pred_len × enc_in.

Args:
    seq_len:     input lookback window.
    pred_len:    forecast horizon.
    enc_in:      number of variates.
    d_model:     Transformer hidden dimension.
    n_heads:     attention heads.
    e_layers:    Transformer encoder layers.
    d_ff:        feed-forward hidden size.
    dropout:     dropout rate.
    dish_hidden: hidden size of the CoSta MLP.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CoSta(nn.Module):
    """Coefficient Statistics network — predicts bias and scale corrections.

    MLP: (B, T, C) → pool over T → (B, C) → MLP → (B, 2 * C)
    Returns phi_B (B, 1, C), phi_W (B, 1, C).
    """

    def __init__(self, seq_len: int, enc_in: int, hidden: int):
        super().__init__()
        # Pool time dim first, then predict coefficients per channel
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * enc_in),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, C) → transpose → (B, C, T) → linear → (B, C, 2) → ...
        # We flatten differently: treat (B, T) as the sequence for channel-wise
        # Actually: mean over C to get (B, T), then linear to predict per-channel

        # Channel-averaged input: (B, T)
        B, T, C = x.shape
        h = x.mean(-1)          # (B, T)
        out = self.net(h)        # (B, 2 * C)
        phi_B = out[:, :C].unsqueeze(1)    # (B, 1, C)
        phi_W = F.softplus(out[:, C:]).unsqueeze(1) + 1e-5   # (B, 1, C) > 0
        return phi_B, phi_W


class _TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.d_k
        q = self.q(x).reshape(B, T, H, Dh).transpose(1, 2)
        k = self.k(x).reshape(B, T, H, Dh).transpose(1, 2)
        v = self.v(x).reshape(B, T, H, Dh).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (Dh ** 0.5)
        attn = F.dropout(F.softmax(scores, dim=-1), p=self.dropout, training=self.training)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        out = self.proj(out)
        x = self.norm1(x + F.dropout(out, p=self.dropout, training=self.training))
        ff = self.ff2(F.gelu(self.ff1(x)))
        ff = F.dropout(ff, p=self.dropout, training=self.training)
        return self.norm2(x + ff)


class DishTS(nn.Module):
    """Dish-TS: input-adaptive distribution shift normalisation + Transformer.

    Two CoSta networks: one predicts the normalisation coefficients for the
    input window, the other predicts de-normalisation coefficients for the
    forecast horizon (both conditioned on the observed input series).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 256,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        dish_hidden: int = 64,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

        # CoSta for input normalisation
        self.costa_in = _CoSta(seq_len, enc_in, dish_hidden)
        # CoSta for forecast de-normalisation
        self.costa_out = _CoSta(seq_len, enc_in, dish_hidden)

        self.input_proj = nn.Linear(enc_in, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList(
            [_TransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, pred_len * enc_in)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)
        Returns:
            (B, pred_len, C)
        """
        B, T, C = x.shape

        # 1. Dish normalisation (input-adaptive)
        phi_B_in, phi_W_in = self.costa_in(x)          # (B, 1, C), (B, 1, C)
        x_norm = (x - phi_B_in) / phi_W_in

        # 2. Forecast de-normalisation coefficients (conditioned on observed x)
        phi_B_out, phi_W_out = self.costa_out(x)

        # 3. Transformer backbone
        h = self.input_proj(x_norm) + self.pos_embed[:, :T, :]
        h = F.dropout(h, p=self.dropout, training=self.training)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        # 4. Project last token to (pred_len, C)
        out = self.output_proj(h[:, -1, :]).reshape(B, self.pred_len, C)

        # 5. Dish de-normalisation
        out = out * phi_W_out + phi_B_out
        return out
