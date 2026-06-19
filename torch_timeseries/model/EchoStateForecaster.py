"""EchoStateForecaster — Echo State Network (reservoir computing) forecaster.

The recurrent reservoir W_res is randomly initialised and NEVER trained.
Only the input projection W_in and the linear readout are learned.
Rich temporal dynamics emerge from the fixed reservoir's echo state property.

    h_t = tanh(W_res · h_{t-1} + W_in · x_t)        # reservoir step
    pred = W_out(mean_pool(h_{1..T}))                 # linear readout

W_res is scaled so spectral_radius(W_res) = target_sr < 1  (stability guarantee).

Reference: Jaeger & Haas, "Harnessing Nonlinearity" Science 2004;
           Lukoševičius & Jaeger, "Reservoir Computing Approaches" 2009.
"""

from __future__ import annotations
import torch
import torch.nn as nn


def _make_reservoir(d_reservoir: int, sparsity: float, spectral_radius: float) -> torch.Tensor:
    """Create a random sparse reservoir matrix scaled to the desired spectral radius."""
    W = torch.randn(d_reservoir, d_reservoir)
    # Apply sparsity mask
    mask = torch.rand(d_reservoir, d_reservoir) < (1.0 - sparsity)
    W = W * mask.float()
    # Scale to target spectral radius
    with torch.no_grad():
        # Use power iteration for large matrices; eigvals for small
        if d_reservoir <= 256:
            eigvals = torch.linalg.eigvals(W)
            sr = eigvals.abs().max().item()
        else:
            # Power iteration approximation
            v = torch.randn(d_reservoir)
            for _ in range(20):
                v = W @ v
                sr_est = v.norm().item()
                v = v / (sr_est + 1e-8)
            sr = sr_est
        if sr > 1e-8:
            W = W * (spectral_radius / sr)
    return W


class EchoStateForecaster(nn.Module):
    """Echo State Network multivariate forecaster.

    Channel-independent (CI) + RevIN.

    Args:
        seq_len:         input sequence length
        pred_len:        forecast horizon
        enc_in:          number of input channels
        d_reservoir:     reservoir hidden size
        sparsity:        fraction of zero connections in W_res (0–1)
        spectral_radius: target spectral radius of W_res (< 1 for stability)
        leak_rate:       leaky integration α: h = (1-α)·h_prev + α·tanh(...)
        dropout:         dropout on readout
        revin:           apply RevIN instance normalisation
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_reservoir: int = 256,
        sparsity: float = 0.9,
        spectral_radius: float = 0.9,
        leak_rate: float = 1.0,
        dropout: float = 0.1,
        revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.leak_rate = leak_rate
        self.revin = revin

        if revin:
            from torch_timeseries.nn.revin import RevIN
            self.rev = RevIN(enc_in)

        # Fixed random reservoir — never trained
        W_res = _make_reservoir(d_reservoir, sparsity, spectral_radius)
        self.register_buffer("W_res", W_res)

        # Trainable input projection and readout
        self.W_in = nn.Linear(1, d_reservoir, bias=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_reservoir, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        if self.revin:
            x = self.rev(x, "norm")

        B, T, C = x.shape
        # CI: (BC, T, 1)
        x_ci = x.permute(0, 2, 1).reshape(B * C, T, 1)

        # Input projection: (BC, T, d_reservoir)
        inp = self.W_in(x_ci)                    # linear, no activation here

        BC = B * C
        D = self.W_res.shape[0]
        h = torch.zeros(BC, D, device=x.device, dtype=x.dtype)

        hiddens = []
        for t in range(T):
            # Reservoir update with leaky integration
            h_new = torch.tanh(h @ self.W_res.T + inp[:, t])
            h = (1.0 - self.leak_rate) * h + self.leak_rate * h_new
            hiddens.append(h)

        # Readout from mean-pooled reservoir states
        reservoir_out = torch.stack(hiddens, dim=1).mean(dim=1)   # (BC, D)
        out = self.head(self.drop(reservoir_out))                  # (BC, pred_len)
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)   # (B, pred_len, C)

        if self.revin:
            out = self.rev(out, "denorm")
        return out
