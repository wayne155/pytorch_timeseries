"""Normalizing Flow Forecaster for probabilistic time-series forecasting.

Uses a **Real-NVP style affine coupling flow** conditioned on a temporal
backbone (VanillaTransformer) to define an exact-likelihood distribution
over the forecast horizon.

Key distinctions from other probabilistic models:
  * **Exact likelihood**: NLL is computed in closed form (unlike MCDropout,
    StudentT, Gaussian which approximate via parametric distributions or
    sampling, unlike Ensemble which has no principled likelihood).
  * **Flexible distribution**: the flow transforms a standard Gaussian into
    an arbitrarily complex multivariate distribution.
  * **Invertible**: can compute exact log-prob of any target y (useful for
    evaluation), and can sample exactly by inverting the flow.

Architecture:
  1. Backbone (VanillaTransformer) encodes the lookback → context vector
     of shape ``(B, pred_len * enc_in)``.
  2. ``K`` affine coupling layers, each splitting the target into two halves:
       - The first half passes through unchanged.
       - The second half is transformed: z₂ = exp(s(z₁, ctx)) ⊙ z₂ + t(z₁, ctx),
         where s, t are small MLPs conditioned on both z₁ and the context.
  3. The transformed z ∈ ℝ^(pred_len × enc_in) can be decoded back to target
     space or sampled from N(0,I) → inverse flow → (B, pred_len, enc_in).

Training: maximize log p(y|x) = log p_z(z) + sum_k log|det J_k|.
Sampling: z ~ N(0,I) → inverse flow → y.

Args:
    seq_len:     input lookback window.
    pred_len:    forecasting horizon.
    enc_in:      number of variates.
    d_model:     transformer hidden size.
    n_heads:     transformer attention heads.
    e_layers:    transformer encoder layers.
    d_ff:        transformer feed-forward size.
    dropout:     dropout in transformer.
    activation:  transformer activation ('relu' or 'gelu').
    revin:       use RevIN normalisation on backbone input.
    flow_layers: number of affine coupling blocks K (default 6).
    flow_hidden: hidden size of s/t MLPs in each coupling block.
    num_samples: default number of samples at inference.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .VanillaTransformer import VanillaTransformer


class _AffineCouplingBlock(nn.Module):
    """One Real-NVP affine coupling layer.

    Splits the target z into (z1, z2), keeps z1 fixed, transforms z2:
        z2' = z2 * exp(s(z1, ctx)) + t(z1, ctx)
    log|det J| = sum(s(...))
    """

    def __init__(self, dim: int, ctx_dim: int, hidden: int = 64):
        super().__init__()
        # dim must be even for a clean 50/50 split
        self.d1 = dim // 2
        self.d2 = dim - self.d1

        inp = self.d1 + ctx_dim
        self.s_net = nn.Sequential(
            nn.Linear(inp, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, self.d2),
        )
        self.t_net = nn.Sequential(
            nn.Linear(inp, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, self.d2),
        )
        # Initialise scale outputs near zero so early training is stable
        nn.init.zeros_(self.s_net[-1].weight)
        nn.init.zeros_(self.s_net[-1].bias)
        nn.init.zeros_(self.t_net[-1].weight)
        nn.init.zeros_(self.t_net[-1].bias)

    def forward(
        self, z: torch.Tensor, ctx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward (data → latent): returns (z', log_det).

        Args:
            z:   ``(B, dim)``
            ctx: ``(B, ctx_dim)``

        Returns:
            z':      ``(B, dim)``
            log_det: ``(B,)``
        """
        z1, z2 = z[:, :self.d1], z[:, self.d1:]
        inp = torch.cat([z1, ctx], dim=-1)
        s = self.s_net(inp)
        t = self.t_net(inp)
        z2_ = z2 * torch.exp(s) + t
        log_det = s.sum(dim=-1)
        return torch.cat([z1, z2_], dim=-1), log_det

    def inverse(self, z: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Inverse (latent → data).

        Args:
            z:   ``(B, dim)``
            ctx: ``(B, ctx_dim)``

        Returns:
            x: ``(B, dim)``
        """
        z1, z2_ = z[:, :self.d1], z[:, self.d1:]
        inp = torch.cat([z1, ctx], dim=-1)
        s = self.s_net(inp)
        t = self.t_net(inp)
        z2 = (z2_ - t) * torch.exp(-s)
        return torch.cat([z1, z2], dim=-1)


class NormalizingFlowForecaster(nn.Module):
    """Conditional Normalizing Flow forecaster.

    Backbone: VanillaTransformer encodes context.
    Flow: K affine coupling layers transform y ↔ z (standard Gaussian).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 256,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        revin: bool = True,
        flow_layers: int = 6,
        flow_hidden: int = 128,
        num_samples: int = 50,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.num_samples = num_samples

        flat_dim = pred_len * enc_in

        # Context encoder
        self.backbone = VanillaTransformer(
            seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
            d_model=d_model, n_heads=n_heads, e_layers=e_layers,
            d_ff=d_ff, dropout=dropout, activation=activation, revin=revin,
        )
        ctx_dim = flat_dim   # backbone output flattened

        # Affine coupling blocks — alternate which half is transformed
        self.flows = nn.ModuleList([
            _AffineCouplingBlock(flat_dim, ctx_dim, flow_hidden)
            for _ in range(flow_layers)
        ])

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #

    def _get_context(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the lookback into a flat context vector."""
        pred = self.backbone(x)           # (B, pred_len, enc_in)
        return pred.reshape(pred.size(0), -1)   # (B, flat_dim)

    # ------------------------------------------------------------------ #
    # flow directions                                                      #
    # ------------------------------------------------------------------ #

    def _forward_flow(
        self, y: torch.Tensor, ctx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map target y → latent z, accumulating log|det J|.

        Args:
            y:   ``(B, flat_dim)``
            ctx: ``(B, ctx_dim)``

        Returns:
            z:       ``(B, flat_dim)``
            log_det: ``(B,)``
        """
        log_det = torch.zeros(y.size(0), device=y.device)
        z = y
        for i, flow in enumerate(self.flows):
            # Alternate which half is transformed by reversing on odd layers
            if i % 2 == 1:
                z = torch.cat([z[:, z.size(1) // 2:], z[:, :z.size(1) // 2]], dim=-1)
            z, ld = flow(z, ctx)
            if i % 2 == 1:
                z = torch.cat([z[:, z.size(1) // 2:], z[:, :z.size(1) // 2]], dim=-1)
            log_det = log_det + ld
        return z, log_det

    def _inverse_flow(self, z: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Map latent z → target y (inverse flow for sampling).

        Args:
            z:   ``(B, flat_dim)``
            ctx: ``(B, ctx_dim)``

        Returns:
            y: ``(B, flat_dim)``
        """
        y = z
        for i, flow in enumerate(reversed(self.flows)):
            layer_idx = len(self.flows) - 1 - i
            if layer_idx % 2 == 1:
                y = torch.cat([y[:, y.size(1) // 2:], y[:, :y.size(1) // 2]], dim=-1)
            y = flow.inverse(y, ctx)
            if layer_idx % 2 == 1:
                y = torch.cat([y[:, y.size(1) // 2:], y[:, :y.size(1) // 2]], dim=-1)
        return y

    # ------------------------------------------------------------------ #
    # training / inference API                                             #
    # ------------------------------------------------------------------ #

    def nll_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of y given x (training objective).

        Args:
            x: ``(B, seq_len, enc_in)``
            y: ``(B, pred_len, enc_in)``

        Returns:
            scalar NLL (to be minimised).
        """
        B = x.size(0)
        flat_dim = self.pred_len * self.enc_in

        ctx = self._get_context(x)              # (B, flat_dim)
        y_flat = y.reshape(B, -1)               # (B, flat_dim)
        z, log_det = self._forward_flow(y_flat, ctx)

        # log p(z) under N(0,I)
        log_pz = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=-1)
        log_py = log_pz + log_det               # (B,)
        return -log_py.mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Point forecast = backbone output (used at evaluation as the mean).

        Args:
            x: ``(B, seq_len, enc_in)``

        Returns:
            ``(B, pred_len, enc_in)``
        """
        return self.backbone(x)

    @torch.no_grad()
    def sample(
        self, x: torch.Tensor, num_samples: Optional[int] = None
    ) -> torch.Tensor:
        """Draw samples from the flow posterior p(y|x).

        Args:
            x:           ``(B, seq_len, enc_in)``
            num_samples: S samples per window.

        Returns:
            ``(B, pred_len, enc_in, S)``
        """
        S = num_samples or self.num_samples
        B = x.size(0)
        flat_dim = self.pred_len * self.enc_in

        ctx = self._get_context(x)              # (B, flat_dim)

        samples = []
        for _ in range(S):
            z = torch.randn(B, flat_dim, device=x.device)
            y_flat = self._inverse_flow(z, ctx)
            y = y_flat.reshape(B, self.pred_len, self.enc_in)
            samples.append(y)

        return torch.stack(samples, dim=-1)     # (B, pred_len, enc_in, S)
