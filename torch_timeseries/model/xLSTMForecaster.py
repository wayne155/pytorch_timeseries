"""xLSTMForecaster — Extended LSTM with matrix memory cell (mLSTM).

mLSTM replaces the scalar cell state with a matrix C ∈ R^{d×d} and uses
exponential (unbounded) gates for input and forget:

    q_t = W_q · x_t,  k_t = W_k · x_t / sqrt(d),  v_t = W_v · x_t
    log_f = W_f · x_t,  log_i = W_i · x_t + b_i
    f_t = exp(log_f_stabilised),  i_t = exp(log_i - m_t)   [numerically stable]
    C_t = f_t · C_t + i_t · (v_t ⊗ k_t)                   [matrix update]
    n_t = f_t · n_t + i_t · k_t                             [normaliser]
    h_t = (C_t @ q_t) / max(|n_t · q_t|, 1)                [retrieval]

Reference: Beck et al., "xLSTM: Extended Long Short-Term Memory" (2024)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _mLSTMCell(nn.Module):
    """Single mLSTM step: matrix memory + exponential gates."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, d_model, bias=True)
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        self.W_f = nn.Linear(d_model, d_model, bias=True)
        self.W_i = nn.Linear(d_model, d_model, bias=True)
        self.scale = math.sqrt(d_model)

    def forward(
        self,
        x: torch.Tensor,
        C: torch.Tensor,
        n: torch.Tensor,
        m: torch.Tensor,
    ):
        """
        x : (B, d_model)
        C : (B, d_model, d_model)  — matrix cell state
        n : (B, d_model)           — normaliser vector
        m : (B, d_model)           — running max for numerical stability
        Returns updated (h, C, n, m)
        """
        q = self.W_q(x)                          # (B, d)
        k = self.W_k(x) / self.scale             # (B, d)
        v = self.W_v(x)                          # (B, d)

        log_f = -F.softplus(-self.W_f(x))        # log forget ≤ 0 → forget ∈ (0,1]
        log_i_raw = self.W_i(x)

        # Numerically stable max-normalisation
        m_new = torch.maximum(log_f + m, log_i_raw)
        f = torch.exp(log_f + m - m_new)         # (B, d)
        i = torch.exp(log_i_raw - m_new)         # (B, d)

        # Outer product update: C += i * (v ⊗ k)
        vk = torch.bmm(v.unsqueeze(2), k.unsqueeze(1))  # (B, d, d)
        C_new = f.unsqueeze(2) * C + i.unsqueeze(2) * vk
        n_new = f * n + i * k

        # Retrieval
        Cq = torch.bmm(C_new, q.unsqueeze(2)).squeeze(2)   # (B, d)
        denom = torch.clamp((n_new * q).sum(dim=-1, keepdim=True).abs(), min=1.0)
        h = Cq / denom

        return h, C_new, n_new, m_new


class _mLSTMBlock(nn.Module):
    """mLSTM cell + LayerNorm + residual, scans over the time axis."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.cell = _mLSTMCell(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, d)
        B, T, D = x_seq.shape
        C = torch.zeros(B, D, D, device=x_seq.device, dtype=x_seq.dtype)
        n = torch.zeros(B, D, device=x_seq.device, dtype=x_seq.dtype)
        m = torch.full((B, D), -1e9, device=x_seq.device, dtype=x_seq.dtype)
        hiddens = []
        for t in range(T):
            h, C, n, m = self.cell(x_seq[:, t], C, n, m)
            hiddens.append(h)
        out = torch.stack(hiddens, dim=1)       # (B, T, d)
        return self.norm(out + x_seq)


class xLSTMForecaster(nn.Module):
    """xLSTM (mLSTM) multivariate forecaster.

    Channel-independent (CI) + RevIN.

    Args:
        seq_len:   input window length
        pred_len:  forecast horizon
        enc_in:    number of input channels
        d_model:   model dimension (= head size = memory dim)
        n_layers:  stacked mLSTM blocks
        dropout:   dropout on the head
        revin:     apply RevIN instance normalisation
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 32,
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
        self.blocks = nn.ModuleList([_mLSTMBlock(d_model) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        if self.revin:
            x = self.rev(x, "norm")

        B, T, C = x.shape
        x_ci = x.permute(0, 2, 1).reshape(B * C, T, 1)
        h = self.embed(x_ci)                    # (BC, T, d)

        for block in self.blocks:
            h = block(h)

        h_last = h[:, -1, :]                    # (BC, d) — last hidden
        out = self.head(self.drop(h_last))      # (BC, pred_len)
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)

        if self.revin:
            out = self.rev(out, "denorm")
        return out
