"""GLAForecaster — Gated Linear Attention forecaster.

GLA (Yang et al., 2024) replaces standard attention's O(T²) dot-product with a
matrix-valued recurrent state whose decay is controlled by *input-dependent* gates:

    S_t = diag(g_t) · S_{t-1} + k_t ⊗ v_t       (outer-product state update)
    o_t = S_t · q_t                                (retrieval via matmul)

where g_t = sigmoid(W_g · x_t) ∈ (0,1)^{d_head} — learned per-step forgetting.

This is architecturally distinct from:
  - LinearAttentionForecaster: no gating on the state
  - RetForecaster:             fixed per-head decay γ^{m-n}, not input-dependent
  - RWKV:                      WKV with scalar max-normalised decay, different structure
  - xLSTM:                     exponential i/f gates, C kept as matrix but different update rule

Reference: Yang et al., "Gated Linear Attention Transformers with Hardware-Efficient
Training", arXiv 2312.06635, 2024.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _GLALayer(nn.Module):
    """Multi-head Gated Linear Attention layer.

    Each head maintains S ∈ R^{d_head × d_head} as its recurrent state.
        S_t[i,j] = g_t[i] · S_{t-1}[i,j] + k_t[i] · v_t[j]   (gated outer product)
        o_t      = S_t @ q_t
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_G = nn.Linear(d_model, d_model, bias=True)   # gate (→ sigmoid)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        Q = self.W_Q(x).view(B, T, H, Dh)          # (B, T, H, Dh)
        K = self.W_K(x).view(B, T, H, Dh)
        V = self.W_V(x).view(B, T, H, Dh)
        G = torch.sigmoid(self.W_G(x)).view(B, T, H, Dh)  # gates ∈ (0,1)

        # Recurrence: S_t = diag(g_t) · S_{t-1} + k_t ⊗ v_t
        # S: (B, H, Dh, Dh)
        S = torch.zeros(B, H, Dh, Dh, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            g_t = G[:, t]           # (B, H, Dh)
            k_t = K[:, t]           # (B, H, Dh)
            v_t = V[:, t]           # (B, H, Dh)
            q_t = Q[:, t]           # (B, H, Dh)

            # Gate rows: S[b, h, i, :] *= g_t[b, h, i]
            S = g_t.unsqueeze(-1) * S + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
            # S += k_t ⊗ v_t: S[b,h,i,j] += k_t[b,h,i] * v_t[b,h,j]

            # Retrieve: o = S @ q
            o_t = torch.einsum("bhij,bhj->bhi", S, q_t)  # (B, H, Dh)
            outs.append(o_t)

        out = torch.stack(outs, dim=1).view(B, T, D)   # (B, T, d_model)
        out = self.drop(self.W_O(out))
        return self.norm(out + x)


class _GLABlock(nn.Module):
    """GLA + GELU-FFN with pre-norm and residual."""

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float) -> None:
        super().__init__()
        self.gla = _GLALayer(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gla(x)
        return x + self.ffn(x)


class GLAForecaster(nn.Module):
    """Gated Linear Attention multivariate forecaster.

    Channel-independent (CI) + RevIN.

    Args:
        seq_len:   input sequence length
        pred_len:  forecast horizon
        enc_in:    number of input channels
        d_model:   model dimension (must be divisible by n_heads)
        n_heads:   number of attention heads
        d_ffn:     feedforward hidden size
        n_layers:  number of GLA blocks
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
        self.blocks = nn.ModuleList([
            _GLABlock(d_model, n_heads, d_ffn, dropout)
            for _ in range(n_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin:
            x = self.rev(x, "norm")

        B, T, C = x.shape
        x_ci = x.permute(0, 2, 1).reshape(B * C, T, 1)
        h = self.embed(x_ci)                        # (BC, T, d)

        for block in self.blocks:
            h = block(h)

        out = self.head(self.drop(h[:, -1]))         # last hidden → (BC, pred_len)
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)

        if self.revin:
            out = self.rev(out, "denorm")
        return out
