"""RWKVForecaster — Receptance-Weighted Key-Value recurrent forecaster.

RWKV reformulates linear attention as a stable recurrence:

    r_t = σ(W_r · x_t)                          # receptance gate
    k_t = W_k · x_t                              # key
    v_t = W_v · x_t                              # value
    w   = -softplus(w_raw)                        # per-dim decay < 0 (learned)
    u   = u_raw                                   # per-dim first-token bonus

    WKV recurrence (max-normalised for stability):
        m_new    = max(m + w, k_t + u)
        A_new    = exp(m + w - m_new)·A + exp(k_t + u - m_new)·v_t
        B_new    = exp(m + w - m_new)·B + exp(k_t + u - m_new)
        wkv_t    = A_new / B_new.clamp(min=1e-6)
        out_t    = r_t · wkv_t

    Then a channel-mixing sub-block (1D feedforward with token-shift):
        out = σ(W_k'·x_mix)·(W_v'·x_mix) + W_r'·x_t  (SwiGLU-style mix)

Reference: Peng et al., "RWKV: Reinventing RNNs for the Transformer Era" (2023)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _TimeFirst(nn.Module):
    """Time-mixing sub-block: WKV recurrence + receptance gate."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        # Learnable token-shift mix ratio μ ∈ (0,1)
        self.mu = nn.Parameter(torch.full((d_model,), 0.5))
        self.W_r = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        # Per-dim time decay (learned, kept < 0 via -softplus)
        self.w_raw = nn.Parameter(torch.ones(d_model))
        # Per-dim first-token bonus
        self.u = nn.Parameter(torch.zeros(d_model))

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, d)
        B, T, D = x_seq.shape
        mu = torch.sigmoid(self.mu)             # per-dim mix in (0,1)
        w = -F.softplus(self.w_raw)             # decay < 0

        # Token-shifted version of x (shift by 1, pad with zeros)
        x_shift = torch.cat([torch.zeros(B, 1, D, device=x_seq.device, dtype=x_seq.dtype),
                              x_seq[:, :-1, :]], dim=1)
        x_mix = mu * x_seq + (1 - mu) * x_shift

        r = torch.sigmoid(self.W_r(x_mix))     # (B, T, d)
        k = self.W_k(x_mix)                    # (B, T, d)
        v = self.W_v(x_mix)                    # (B, T, d)

        # WKV recurrence (max-normalised)
        A = torch.zeros(B, D, device=x_seq.device, dtype=x_seq.dtype)
        M = torch.zeros(B, D, device=x_seq.device, dtype=x_seq.dtype)
        B_den = torch.zeros(B, D, device=x_seq.device, dtype=x_seq.dtype)
        wkvs = []
        for t in range(T):
            kt = k[:, t]                        # (B, d)
            vt = v[:, t]                        # (B, d)
            m_new = torch.maximum(M + w, kt + self.u)
            eA = torch.exp(M + w - m_new)
            eB = torch.exp(kt + self.u - m_new)
            A = eA * A + eB * vt
            B_den = eA * B_den + eB
            M = m_new
            wkv = A / B_den.clamp(min=1e-6)
            wkvs.append(wkv)

        wkv_seq = torch.stack(wkvs, dim=1)     # (B, T, d)
        out = r * wkv_seq
        return self.W_o(out)


class _ChannelMix(nn.Module):
    """Channel-mixing sub-block: token-shifted SwiGLU feedforward."""

    def __init__(self, d_model: int, d_ffn: int) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.full((d_model,), 0.5))
        self.W_k = nn.Linear(d_model, d_ffn, bias=False)
        self.W_v = nn.Linear(d_ffn, d_model, bias=False)
        self.W_r = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, T, D = x_seq.shape
        mu = torch.sigmoid(self.mu)
        x_shift = torch.cat([torch.zeros(B, 1, D, device=x_seq.device, dtype=x_seq.dtype),
                              x_seq[:, :-1, :]], dim=1)
        x_mix = mu * x_seq + (1 - mu) * x_shift
        k_out = torch.relu(self.W_k(x_mix)) ** 2   # squared-ReLU
        return torch.sigmoid(self.W_r(x_mix)) * self.W_v(k_out)


class _RWKVBlock(nn.Module):
    """One RWKV block: TimeFirst + LayerNorm + ChannelMix + LayerNorm."""

    def __init__(self, d_model: int, d_ffn: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.time_mix = _TimeFirst(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.chan_mix = _ChannelMix(d_model, d_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.time_mix(self.norm1(x))
        x = x + self.chan_mix(self.norm2(x))
        return x


class RWKVForecaster(nn.Module):
    """RWKV multivariate forecaster.

    Channel-independent (CI) + RevIN.

    Args:
        seq_len:   input sequence length
        pred_len:  forecast horizon
        enc_in:    number of input channels
        d_model:   model dimension
        d_ffn:     channel-mix hidden dimension (default 4×d_model)
        n_layers:  number of RWKV blocks
        dropout:   dropout on head
        revin:     apply RevIN instance normalisation
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        d_ffn: int | None = None,
        n_layers: int = 2,
        dropout: float = 0.1,
        revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.revin = revin
        d_ffn = d_ffn or d_model * 4

        if revin:
            from torch_timeseries.nn.revin import RevIN
            self.rev = RevIN(enc_in)

        self.embed = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList(
            [_RWKVBlock(d_model, d_ffn) for _ in range(n_layers)]
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin:
            x = self.rev(x, "norm")

        B, T, C = x.shape
        x_ci = x.permute(0, 2, 1).reshape(B * C, T, 1)
        h = self.embed(x_ci)                # (BC, T, d)

        for block in self.blocks:
            h = block(h)

        h = self.norm_out(h[:, -1, :])      # last position
        out = self.head(self.drop(h))       # (BC, pred_len)
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)

        if self.revin:
            out = self.rev(out, "denorm")
        return out
