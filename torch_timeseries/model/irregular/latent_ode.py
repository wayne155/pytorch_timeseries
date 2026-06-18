"""Latent ODE for irregular time series.

Requires: pip install torch-timeseries[irregular]  (installs torchdiffeq)
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class _ODEFunc(nn.Module):
    """Simple MLP ODE function: dz/dt = f(z)."""
    def __init__(self, latent_size: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, latent_size),
        )

    def forward(self, t, z):
        return self.net(z)


class LatentODE(nn.Module):
    """Variational Latent ODE for irregular time series.

    Requires ``torchdiffeq``::

        pip install torch-timeseries[irregular]

    Architecture:
      1. RNN encoder over reversed observations → (mu, logvar) for z0.
      2. Sample z0 ~ N(mu, exp(0.5 * logvar)).
      3. ODE solver from t=0 to t=1 via z0, queried at t_query.
      4. Linear decoder: z(t) → output.

    - ``forward(x, t, mask)`` → ``(B, output_size)`` for classification.
    - ``forward(x, t, mask, t_query=...)`` → ``(B, Tq, output_size)`` for seq2seq.
    """

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        hidden_size: int,
        output_size: int,
        ode_method: str = "dopri5",
    ) -> None:
        try:
            from torchdiffeq import odeint  # noqa: F401
        except ImportError:
            raise ImportError(
                "LatentODE requires torchdiffeq. "
                "Install it with: pip install torch-timeseries[irregular]"
            )
        super().__init__()
        self.latent_size = latent_size
        self.ode_method = ode_method

        self.encoder_rnn = nn.GRU(input_size * 2, hidden_size, batch_first=True)
        self.z0_proj = nn.Linear(hidden_size, latent_size * 2)   # mu + logvar
        self.ode_func = _ODEFunc(latent_size, hidden_size)
        self.decoder = nn.Linear(latent_size, output_size)
        self.fc_cls = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def _encode(self, x: Tensor, mask: Tensor):
        x_in = torch.cat([x, mask], dim=-1)          # (B, T, F*2)
        x_rev = torch.flip(x_in, dims=[1])
        _, h = self.encoder_rnn(x_rev)
        h = h.squeeze(0)                             # (B, H)
        z0_params = self.z0_proj(h)
        return z0_params.chunk(2, dim=-1)            # mu, logvar

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        mask: Tensor,
        x_time: Tensor = None,
        t_query: Tensor = None,
    ) -> Tensor:
        from torchdiffeq import odeint

        mu, logvar = self._encode(x, mask)
        z0 = self._reparameterize(mu, logvar)        # (B, latent)

        if t_query is None:
            return self.fc_cls(z0)

        B, Tq = t_query.shape
        t_grid = torch.linspace(0.0, 1.0, Tq + 1, device=x.device)
        z_traj = odeint(self.ode_func, z0, t_grid, method=self.ode_method)
        z_at_query = z_traj[1:].permute(1, 0, 2)    # (B, Tq, latent)
        return self.decoder(z_at_query)
