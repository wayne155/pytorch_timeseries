"""ImplicitNeuralForecaster: Implicit Neural Representation (INR) for time series.

Key idea:

Model the forecast as a *continuous function* of time, parameterised by a
coordinate MLP (implicit neural representation, NeRF-style):

    f(c, t) → y_t

where c ∈ ℝ^{d_latent} is a context vector encoding the observed history and
t ∈ [0, 1] is the normalised output time coordinate.

Training loop (per sample):
    1. Encode the input window x ∈ ℝ^T into context c via an encoder MLP.
    2. For each output time step s ∈ {1, …, H}:
           embed  t_s = s / H  →  time embedding γ(t_s) ∈ ℝ^{d_time}
           decode y_s = INR_MLP(cat[c, γ(t_s)])  ∈ ℝ

All H decode calls are batched: the INR_MLP receives (B·C, H, d_latent+d_time)
and produces (B·C, H) at once.

Time embedding (Fourier/sinusoidal):
    γ(t) = [1, sin(2πt), cos(2πt), sin(4πt), cos(4πt), …]  ∈ ℝ^{2K+1}

This contrasts with every existing model in the library:
  - Sequential models (RNN, Transformer, etc.): produce H steps sequentially.
  - Direct models (DLinear, MLP, etc.): map x → ŷ as a single dense vector.
  - ImplicitNeuralForecaster: each step is *independently decoded* from (c, t_s).

Architecture::

    Input (B, T, C)
      → RevIN norm
      → CI reshape → (B·C, T)
      → EncoderMLP: T → d_latent  (latent context per channel)
      → expand to (B·C, H, d_latent)
      → cat with time embeddings (B·C, H, d_latent + d_time)
      → INR_MLP → (B·C, H, 1) → squeeze
      → reshape → (B, pred_len, C)
      → RevIN denorm

Args:
    seq_len:    input lookback T.
    pred_len:   forecast horizon H.
    enc_in:     number of variates C.
    d_latent:   context vector dimension.
    d_time:     time embedding width (2K+1 sinusoidal features).
    enc_layers: depth of the encoder MLP.
    dec_layers: depth of the INR decoder MLP.
    d_hidden:   hidden width for both MLPs.
    dropout:    dropout probability.
    revin:      use RevIN instance normalisation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.revin import RevIN


def _time_embedding(n_steps: int, d_time: int, device=None) -> torch.Tensor:
    """Build (n_steps, d_time) sinusoidal time embedding.

    d_time = 2K+1 where K = (d_time-1)//2.
    """
    t = torch.linspace(0, 1, n_steps, device=device)    # (H,)
    K = (d_time - 1) // 2
    feats = [torch.ones_like(t)]
    for k in range(1, K + 1):
        feats.append(torch.sin(2 * math.pi * k * t))
        feats.append(torch.cos(2 * math.pi * k * t))
    emb = torch.stack(feats[:d_time], dim=-1)            # (H, d_time)
    return emb


def _make_mlp(in_dim: int, out_dim: int, d_hidden: int, n_layers: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for i in range(n_layers - 1):
        layers += [nn.Linear(dim, d_hidden), nn.GELU(), nn.Dropout(dropout)]
        dim = d_hidden
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class ImplicitNeuralForecaster(nn.Module):
    """INR-based forecaster: context encoder + coordinate-MLP decoder (CI)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_latent: int = 64,
        d_time: int = 33,          # 2K+1 sinusoidal features (K=16)
        enc_layers: int = 2,
        dec_layers: int = 3,
        d_hidden: int = 128,
        dropout: float = 0.1,
        revin: bool = True,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in
        self._revin_flag = revin
        self.d_time = d_time

        if revin:
            self.revin_layer = RevIN(enc_in, affine=True)

        # Encoder MLP: T → d_latent
        self.encoder = _make_mlp(seq_len, d_latent, d_hidden, enc_layers, dropout)

        # INR decoder: (d_latent + d_time) → 1
        self.decoder = _make_mlp(d_latent + d_time, 1, d_hidden, dec_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, pred_len, C)"""
        B, T, C = x.shape
        H = self.pred_len

        if self._revin_flag:
            x = self.revin_layer(x, "norm")

        x_ci = x.permute(0, 2, 1).reshape(B * C, T)         # (BC, T)

        # Encode history to context vector
        ctx = self.encoder(x_ci)                             # (BC, d_latent)

        # Build time embeddings for all output steps
        t_emb = _time_embedding(H, self.d_time, device=x.device)  # (H, d_time)

        # Expand for batch: (BC, H, d_latent) and (BC, H, d_time)
        ctx_exp = ctx.unsqueeze(1).expand(-1, H, -1)         # (BC, H, d_latent)
        t_exp = t_emb.unsqueeze(0).expand(B * C, -1, -1)    # (BC, H, d_time)

        # Decode: coordinate MLP processes each (context, t) pair
        inp = torch.cat([ctx_exp, t_exp], dim=-1)            # (BC, H, d_latent+d_time)
        out = self.decoder(inp).squeeze(-1)                  # (BC, H)

        pred = out.reshape(B, C, H).permute(0, 2, 1)        # (B, H, C)

        if self._revin_flag:
            pred = self.revin_layer(pred, "denorm")
        return pred
