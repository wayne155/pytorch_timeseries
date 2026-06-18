"""Student-T distributional forecasting head.

Extends the Gaussian heteroscedastic head with a learned degrees-of-freedom
parameter ν, making the predictive distribution heavier-tailed.  As ν → ∞
the Student-T converges to a Gaussian; small ν captures fat tails present
in financial and meteorological data.

Reference: Salinas et al., "DeepAR: Probabilistic Forecasting with
Autoregressive Recurrent Networks", Int. Journal of Forecasting 2020.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.distributions as D


class StudentTForecaster(nn.Module):
    """Attach a heteroscedastic Student-T head to a point-forecasting backbone.

    The backbone is expected to produce ``(B, pred_len, enc_in)`` outputs.
    Two additional heads share the backbone's output to learn per-output
    log-standard-deviation and log-degrees-of-freedom.

    Args:
        backbone: deterministic ``nn.Module`` with signature
            ``forward(x) -> (B, pred_len, enc_in)``.
        enc_in: number of output channels (features).
        pred_len: forecasting horizon.
        num_samples: default ensemble size *S* used by :meth:`sample`.
        min_log_sigma: clamp ``log_sigma`` from below.
        max_log_sigma: clamp ``log_sigma`` from above.
        min_log_nu: clamp ``log_nu`` from below (log(2) ≈ 0.69 → ν=2).
        max_log_nu: clamp ``log_nu`` from above (log(30) ≈ 3.4 → ν≈30,
            essentially Gaussian).
    """

    def __init__(
        self,
        backbone: nn.Module,
        enc_in: int,
        pred_len: int,
        num_samples: int = 50,
        min_log_sigma: float = -10.0,
        max_log_sigma: float = 2.0,
        min_log_nu: float = 0.69,   # log(2) → ν=2 (minimum for finite variance)
        max_log_nu: float = 3.5,    # log(33) → near-Gaussian
    ):
        super().__init__()
        self.backbone = backbone
        self.enc_in = enc_in
        self.pred_len = pred_len
        self.num_samples = num_samples
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        self.min_log_nu = min_log_nu
        self.max_log_nu = max_log_nu

        flat_dim = enc_in * pred_len
        self.log_sigma_head = nn.Linear(flat_dim, flat_dim)
        self.log_nu_head = nn.Linear(flat_dim, flat_dim)

        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.constant_(self.log_sigma_head.bias, -1.0)   # σ ≈ 0.37 initially
        nn.init.zeros_(self.log_nu_head.weight)
        nn.init.constant_(self.log_nu_head.bias, 2.0)        # ν ≈ e² ≈ 7 initially

    # ------------------------------------------------------------------ #
    # forward helpers                                                     #
    # ------------------------------------------------------------------ #

    def _split(self, x: torch.Tensor):
        """Return ``(mu, log_sigma, log_nu)`` each of shape ``(B, O, N)``."""
        mu = self.backbone(x)
        flat = mu.flatten(1)
        log_sigma = self.log_sigma_head(flat).view_as(mu).clamp(
            self.min_log_sigma, self.max_log_sigma
        )
        log_nu = self.log_nu_head(flat).view_as(mu).clamp(
            self.min_log_nu, self.max_log_nu
        )
        return mu, log_sigma, log_nu

    # ------------------------------------------------------------------ #
    # public API                                                          #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor):
        """Return ``(mu, log_sigma, log_nu)`` used during training."""
        return self._split(x)

    def nll_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Student-T NLL loss averaged over all outputs.

        Args:
            x: input ``(B, seq_len, enc_in)``.
            y: target ``(B, pred_len, enc_in)``.
        """
        mu, log_sigma, log_nu = self._split(x)
        nu = log_nu.exp()
        sigma = log_sigma.exp()
        dist = D.StudentT(df=nu, loc=mu, scale=sigma)
        return -dist.log_prob(y).mean()

    @torch.no_grad()
    def sample(
        self, x: torch.Tensor, num_samples: Optional[int] = None
    ) -> torch.Tensor:
        """Sample from the Student-T predictive distribution.

        Args:
            x: input batch ``(B, seq_len, enc_in)``.
            num_samples: override for ``self.num_samples``.

        Returns:
            Tensor ``(B, pred_len, enc_in, S)`` — samples from StudentT(ν, μ, σ).
        """
        S = num_samples if num_samples is not None else self.num_samples
        mu, log_sigma, log_nu = self._split(x)
        nu = log_nu.exp()
        sigma = log_sigma.exp()
        dist = D.StudentT(df=nu, loc=mu, scale=sigma)
        # rsample is not available for StudentT → use sample() in a loop
        samples = torch.stack(
            [dist.sample() for _ in range(S)], dim=-1
        )
        return samples  # (B, O, N, S)
