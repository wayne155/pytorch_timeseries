"""SpikeForecaster — Spiking Neural Network (LIF) multivariate forecaster.

Leaky Integrate-and-Fire (LIF) neurons with smooth surrogate gradients:

    V_t = α · V_{t-1} + (1-α) · (W_x · x_t)      # membrane potential
    s_t = sigmoid((V_t - θ) / τ)                   # soft spike (surrogate)
    α   = sigmoid(α_raw)                            # learned membrane decay ∈ (0,1)

Stacked LIF layers with skip connections; final mean-pooled spike rate → head.
Surrogate gradient avoids the dead-neuron problem of hard Heaviside thresholds.

Reference: Maass 1997 (LIF neurons); Neftci et al. 2019 (surrogate gradient SNN)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _LIFLayer(nn.Module):
    """One LIF (Leaky Integrate-and-Fire) recurrent layer.

    Membrane: V_t = α · V_{t-1} + (1-α) · linear(x_t)
    Spike    : s_t = sigmoid((V_t - θ) / τ)
    """

    def __init__(self, in_dim: int, out_dim: int, surrogate_tau: float = 0.5) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.tau = surrogate_tau
        # Input → membrane projection
        self.W_x = nn.Linear(in_dim, out_dim, bias=True)
        # Learnable membrane decay α (unconstrained → sigmoid)
        self.alpha_raw = nn.Parameter(torch.zeros(out_dim))
        # Learnable firing threshold θ
        self.theta = nn.Parameter(torch.ones(out_dim) * 0.5)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, in_dim) → returns (B, T, out_dim)
        B, T, _ = x_seq.shape
        alpha = torch.sigmoid(self.alpha_raw)   # (out_dim,)
        V = torch.zeros(B, self.out_dim, device=x_seq.device, dtype=x_seq.dtype)

        spikes = []
        for t in range(T):
            # Membrane update: leaky integration
            inp = self.W_x(x_seq[:, t])                         # (B, out)
            V = alpha * V + (1 - alpha) * inp
            # Soft spike
            s = torch.sigmoid((V - self.theta) / self.tau)      # (B, out)
            spikes.append(s)

        return torch.stack(spikes, dim=1)                        # (B, T, out_dim)


class _LIFBlock(nn.Module):
    """LIF layer + LayerNorm + residual projection."""

    def __init__(self, d_model: int, surrogate_tau: float) -> None:
        super().__init__()
        self.lif = _LIFLayer(d_model, d_model, surrogate_tau)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        s = self.lif(x)                         # (B, T, d)
        return self.norm(s + x)                 # residual


class SpikeForecaster(nn.Module):
    """Spiking Neural Network (LIF) multivariate forecaster.

    Channel-independent (CI) + RevIN.

    Args:
        seq_len:        input sequence length
        pred_len:        forecast horizon
        enc_in:          number of input channels
        d_model:         hidden dimension
        n_layers:        number of LIF blocks
        surrogate_tau:   sigmoid temperature for surrogate gradient
        dropout:         dropout on head
        revin:           apply RevIN instance normalisation
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        n_layers: int = 2,
        surrogate_tau: float = 0.5,
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
            [_LIFBlock(d_model, surrogate_tau) for _ in range(n_layers)]
        )
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

        # Mean-pooled spike rate → prediction
        out = self.head(self.drop(h.mean(dim=1)))     # (BC, pred_len)
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)

        if self.revin:
            out = self.rev(out, "denorm")
        return out
