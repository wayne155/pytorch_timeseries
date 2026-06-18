"""Neural Controlled Differential Equation for irregular time series.

Requires: pip install torch-timeseries[irregular]  (installs torchcde)
Only supports classification (terminal hidden state).
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor


class _CDEFunc(nn.Module):
    """CDE vector field: dz/dt = f(z) * dX/dt."""
    def __init__(self, input_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_channels = input_channels
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size * input_channels),
        )

    def forward(self, t, z):
        out = self.net(z)                                         # (B, H*C)
        return out.view(z.shape[0], self.hidden_size, self.input_channels)


class NeuralCDE(nn.Module):
    """Neural CDE for irregular time-series classification.

    Requires ``torchcde``::

        pip install torch-timeseries[irregular]

    Fits a natural cubic spline to the irregular observations and drives
    a CDE with that spline. Returns ``(B, output_size)`` logits.
    Only supports classification (not seq2seq).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        interpolation: str = "cubic",
    ) -> None:
        try:
            import torchcde  # noqa: F401
        except ImportError:
            raise ImportError(
                "NeuralCDE requires torchcde. "
                "Install it with: pip install torch-timeseries[irregular]"
            )
        super().__init__()
        self.interpolation = interpolation
        self.input_size = input_size
        self.hidden_size = hidden_size

        # input_channels = F + 1 (features + time channel)
        self.cde_func = _CDEFunc(input_size + 1, hidden_size)
        self.initial_proj = nn.Linear(input_size + 1, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        mask: Tensor,
        x_time: Tensor = None,
        t_query: Tensor = None,
    ) -> Tensor:
        import torchcde

        B, T, F = x.shape
        t_expand = t.unsqueeze(-1)                   # (B, T, 1)
        X = torch.cat([t_expand, x], dim=-1)         # (B, T, F+1)

        coeffs = torchcde.natural_cubic_coeffs(X)
        X_spline = torchcde.NaturalCubicSpline(coeffs)

        z0 = self.initial_proj(X[:, 0, :])           # (B, H)
        z_T = torchcde.cdeint(
            X=X_spline, func=self.cde_func, z0=z0,
            t=t[0], method="rk4",
        )
        return self.fc(z_T[:, -1])                   # (B, output_size)
