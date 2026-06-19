"""AFTForecaster — Attention-Free Transformer forecaster.

AFT replaces the QK dot-product with a learned per-pair position bias w_ts:

    output_t = σ(Q_t) ⊙ Σ_s softmax_s(w_ts + K_s) ⊙ V_s

where w ∈ R^{T×T} is a learned (not data-dependent) position-weight matrix and
the softmax is computed over the s dimension (causal masking applied).

Unlike LinearAttentionForecaster (kernel approx) or SparseTransformerForecaster
(sparse pattern), AFT is exact but data-independent in its mixing weights.

Reference: Zhai et al., "An Attention Free Transformer" (2021)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _AFTBlock(nn.Module):
    """AFT self-attention block with causal learned position bias."""

    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        # Learned position bias: w[t, s] scores how much position s attends to t
        self.w = nn.Parameter(torch.zeros(seq_len, seq_len))
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        B, T, D = x.shape
        Q = torch.sigmoid(self.W_Q(x))    # (B, T, d) — receptance gate
        K = self.W_K(x)                   # (B, T, d) — key
        V = self.W_V(x)                   # (B, T, d) — value

        # Position bias (sliced to T in case seq_len > T)
        w_mat = self.w[:T, :T]            # (T_out, T_in)

        # Broadcast: w_mat(1,T,T,1) + K(B,1,T,d) → (B, T_out, T_in, d)
        wK = w_mat[None, :, :, None] + K[:, None, :, :]

        # Causal mask: only attend to positions s <= t
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        wK = wK.masked_fill(~causal[None, :, :, None], float("-inf"))

        # Per-output-position softmax over the T_in dimension (dim=2)
        attn = F.softmax(wK, dim=2)       # (B, T_out, T_in, d)

        # Weighted sum of V: Σ_s attn[b,t,s,d] * V[b,s,d] → (B, T, d)
        out = (attn * V[:, None, :, :]).sum(dim=2)

        # Receptance gate
        out = Q * out                     # (B, T, d)
        out = self.drop(self.W_O(out))

        return self.norm(out + x)


class AFTForecaster(nn.Module):
    """Attention-Free Transformer multivariate forecaster.

    Channel-independent (CI) + RevIN.

    Args:
        seq_len:   input sequence length
        pred_len:  forecast horizon
        enc_in:    number of input channels
        d_model:   model (embedding) dimension
        n_layers:  number of AFT blocks
        dropout:   dropout rate
        revin:     apply RevIN instance normalisation
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.revin = revin

        if revin:
            from torch_timeseries.nn.revin import RevIN
            self.rev = RevIN(enc_in)

        self.embed = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList(
            [_AFTBlock(d_model, seq_len, dropout) for _ in range(n_layers)]
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        if self.revin:
            x = self.rev(x, "norm")

        B, T, C = x.shape
        x_ci = x.permute(0, 2, 1).reshape(B * C, T, 1)
        h = self.embed(x_ci)              # (BC, T, d)

        for block in self.blocks:
            h = block(h)

        # Mean-pool over time → head
        out = self.head(self.drop(h.mean(dim=1)))     # (BC, pred_len)
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)

        if self.revin:
            out = self.rev(out, "denorm")
        return out
