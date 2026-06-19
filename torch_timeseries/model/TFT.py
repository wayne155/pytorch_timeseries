"""TFT: Temporal Fusion Transformer for multi-horizon time-series forecasting.

Reference: Lim et al., "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting", NeurIPS 2021 (Google DeepMind).

Key ideas:
  TFT combines recurrent processing, variable selection, and attention in a
  single architecture designed for multi-horizon forecasting:

    1. **Gated Residual Network (GRN)**: core building block that applies a
       two-layer dense network with ELU activation, gating (GLU), and a
       residual connection with layer-norm.  Optionally conditioned on a
       context vector.

    2. **Variable Selection Network (VSN)**: softmax-weighted mix of
       per-variable GRNs, producing a single fused embedding per time step.
       Allows the model to focus on informative inputs.

    3. **LSTM encoder-decoder**: sequence-to-sequence LSTM that produces
       context-enriched representations for both past and future steps.

    4. **Temporal self-attention**: scaled dot-product attention on the full
       (past + future) sequence to capture long-range dependencies.

    5. **Point-wise feed-forward** with gating + residual.

    6. **Quantile output**: in the standard paper multiple quantiles are
       predicted; here we output the mean prediction (pred_len × enc_in).

Simplified implementation differences from the paper:
  - Single input type (past observed, no static or known future covariates).
  - No static context conditioning.
  - Single LSTM encoder (no separate encoder/decoder LSTMs for static context).
  - Decoder is replaced by repeating the last LSTM hidden state for pred_len
    steps and refining with temporal attention over the encoder outputs.

Args:
    seq_len:    input lookback window.
    pred_len:   forecast horizon.
    enc_in:     number of variates.
    d_model:    hidden dimension (all layers use this size).
    n_heads:    attention heads.
    num_lstm_layers: number of stacked LSTM layers.
    dropout:    dropout rate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GatedLinearUnit(nn.Module):
    """Split last dim in half; apply sigmoid gate to second half."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size * 2)
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        val, gate = out[..., : self.output_size], out[..., self.output_size :]
        return val * torch.sigmoid(gate)


class _GRN(nn.Module):
    """Gated Residual Network.

    out = LayerNorm(residual(x) + GLU(ELU(W2(ELU(W1(x) + [ctx])))))
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 context_size: int = 0, dropout: float = 0.1):
        super().__init__()
        self.skip = (
            nn.Linear(input_size, output_size)
            if input_size != output_size else nn.Identity()
        )
        self.fc1 = nn.Linear(input_size + context_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate = _GatedLinearUnit(hidden_size, output_size)
        self.norm = nn.LayerNorm(output_size)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        inp = x if context is None else torch.cat([x, context], dim=-1)
        h = F.elu(self.fc1(inp))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.fc2(h))
        h = self.gate(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.norm(self.skip(x) + h)


class _VariableSelectionNetwork(nn.Module):
    """Softmax-weighted mixture of per-variable GRNs.

    input: (B, T, C)  (C variates, each scalar)
    output: (B, T, d_model) fused representation
    """

    def __init__(self, enc_in: int, d_model: int, dropout: float):
        super().__init__()
        # one GRN per variate (scalar embedding)
        self.var_grns = nn.ModuleList(
            [_GRN(1, d_model, d_model, dropout=dropout) for _ in range(enc_in)]
        )
        # selection network: flatten all variates → softmax weights
        self.select_grn = _GRN(enc_in, d_model, enc_in, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        # per-variate embeddings: list of (B, T, d_model)
        var_embs = torch.stack(
            [grn(x[..., i : i + 1]) for i, grn in enumerate(self.var_grns)],
            dim=-2,
        )  # (B, T, C, d_model)

        # selection weights
        weights = F.softmax(self.select_grn(x), dim=-1)  # (B, T, C)
        # weighted sum over variates
        fused = (var_embs * weights.unsqueeze(-1)).sum(dim=-2)  # (B, T, d_model)
        return fused


class _TemporalSelfAttention(nn.Module):
    """Multi-head self-attention with gated skip."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.d_k

        q = self.q_proj(x).reshape(B, T, H, Dh).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (Dh ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)


class TFT(nn.Module):
    """Temporal Fusion Transformer (simplified encoder-only variant).

    Pipeline:
        x → VSN → LSTM encoder → temporal self-attention → GRN → output proj
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model

        # 1. instance normalisation (RevIN-style, no learnable params here)
        # 2. variable selection
        self.vsn = _VariableSelectionNetwork(enc_in, d_model, dropout)

        # 3. LSTM encoder
        self.lstm = nn.LSTM(
            d_model, d_model, num_layers=num_lstm_layers,
            batch_first=True, dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # 4. GLU gating on LSTM output
        self.lstm_gate = nn.Sequential(
            _GatedLinearUnit(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # 5. static enrichment GRN (acts as positional enrichment here)
        self.enrich_grn = _GRN(d_model, d_model, d_model, dropout=dropout)

        # 6. temporal self-attention
        self.attn = _TemporalSelfAttention(d_model, n_heads, dropout)
        self.attn_gate = nn.Sequential(
            _GatedLinearUnit(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # 7. point-wise feed-forward
        self.ff_grn = _GRN(d_model, d_model * 4, d_model, dropout=dropout)

        # 8. output: project pred_len decoder tokens to enc_in
        self.output_proj = nn.Linear(d_model, enc_in)

        # 9. decoder query: learned positional embeddings for pred_len steps
        self.decoder_pos = nn.Parameter(torch.zeros(1, pred_len, d_model))
        nn.init.trunc_normal_(self.decoder_pos, std=0.02)

        self.dropout = dropout

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)
        Returns:
            (B, pred_len, C)
        """
        B, T, C = x.shape

        # instance normalisation
        mu = x.mean(1, keepdim=True)
        sigma = x.std(1, keepdim=True).clamp(1e-5)
        x_norm = (x - mu) / sigma

        # variable selection → (B, T, d_model)
        vsn_out = self.vsn(x_norm)

        # LSTM encoder
        lstm_out, (h_n, c_n) = self.lstm(vsn_out)   # lstm_out: (B, T, d)

        # gated residual around LSTM
        lstm_out = self.lstm_gate(lstm_out) + vsn_out

        # static enrichment
        enriched = self.enrich_grn(lstm_out)          # (B, T, d)

        # build decoder queries: repeat last h_n + learned positional bias
        # h_n: (num_layers, B, d) → take last layer → (B, d)
        ctx = h_n[-1].unsqueeze(1).expand(-1, self.pred_len, -1)  # (B, pred_len, d)
        decoder_q = ctx + self.decoder_pos                          # (B, pred_len, d)

        # concatenate encoder output + decoder queries for full-sequence attention
        full_seq = torch.cat([enriched, decoder_q], dim=1)   # (B, T+pred_len, d)

        # temporal self-attention
        attn_out = self.attn(full_seq)
        attn_out = self.attn_gate(attn_out) + full_seq

        # take decoder portion
        dec_out = attn_out[:, T:, :]                          # (B, pred_len, d)

        # feed-forward
        dec_out = self.ff_grn(dec_out)

        # project to enc_in
        out = self.output_proj(dec_out)                       # (B, pred_len, C)

        # de-normalise
        out = out * sigma + mu
        return out
