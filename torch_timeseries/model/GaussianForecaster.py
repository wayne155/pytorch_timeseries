"""Gaussian distributional forecasting head.

Wraps any point-forecasting backbone and adds a learned log-standard-deviation
head.  Training minimises the Gaussian NLL; inference samples from the
resulting distribution to produce an ensemble compatible with the existing
``(B, pred_len, N, S)`` probabilistic metrics.

Reference: Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep
Learning for Computer Vision?", NeurIPS 2017.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianForecaster(nn.Module):
    """Attach a heteroscedastic Gaussian head to a point-forecasting backbone.

    The backbone is expected to produce ``(B, pred_len, enc_in)`` outputs.  A
    small learned ``log_sigma`` head (initialised to predict near-zero log-std)
    shares the backbone's hidden representation and outputs the same shape.

    Args:
        backbone: deterministic ``nn.Module`` with signature
            ``forward(x) -> (B, pred_len, enc_in)``.
        enc_in: number of output channels (features).
        pred_len: forecasting horizon.
        num_samples: default ensemble size *S* used by :meth:`sample`.
        min_log_sigma: clamp ``log_sigma`` from below for numerical stability.
        max_log_sigma: clamp ``log_sigma`` from above.
    """

    def __init__(
        self,
        backbone: nn.Module,
        enc_in: int,
        pred_len: int,
        num_samples: int = 50,
        min_log_sigma: float = -10.0,
        max_log_sigma: float = 2.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.enc_in = enc_in
        self.pred_len = pred_len
        self.num_samples = num_samples
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        # Small head that emits per-output log-standard-deviation.
        self.log_sigma_head = nn.Linear(enc_in * pred_len, enc_in * pred_len)
        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.constant_(self.log_sigma_head.bias, -1.0)  # start near σ=e^{-1}≈0.37

    # ------------------------------------------------------------------ #
    # forward helpers                                                     #
    # ------------------------------------------------------------------ #

    def _split(self, x: torch.Tensor):
        """Return (mu, log_sigma) of shape (B, pred_len, enc_in) each."""
        mu = self.backbone(x)                                        # (B, O, N)
        flat = mu.flatten(1)                                         # (B, O*N)
        log_sigma = self.log_sigma_head(flat).view_as(mu)           # (B, O, N)
        log_sigma = log_sigma.clamp(self.min_log_sigma, self.max_log_sigma)
        return mu, log_sigma

    # ------------------------------------------------------------------ #
    # public API                                                          #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor):
        """Return ``(mu, log_sigma)`` used during training.

        Returns:
            mu:        ``(B, pred_len, enc_in)``
            log_sigma: ``(B, pred_len, enc_in)``
        """
        return self._split(x)

    def nll_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gaussian NLL loss averaged over all outputs.

        Args:
            x: input ``(B, seq_len, enc_in)``.
            y: target ``(B, pred_len, enc_in)``.

        Returns:
            Scalar loss tensor.
        """
        mu, log_sigma = self._split(x)
        sigma2 = (2 * log_sigma).exp()
        nll = 0.5 * (((y - mu) ** 2) / sigma2 + 2 * log_sigma + math.log(2 * math.pi))
        return nll.mean()

    @torch.no_grad()
    def sample(
        self, x: torch.Tensor, num_samples: Optional[int] = None
    ) -> torch.Tensor:
        """Sample from the Gaussian predictive distribution.

        Args:
            x: input batch ``(B, seq_len, enc_in)``.
            num_samples: override for ``self.num_samples``.

        Returns:
            Tensor ``(B, pred_len, enc_in, S)`` — samples from N(μ, σ²).
        """
        S = num_samples if num_samples is not None else self.num_samples
        mu, log_sigma = self._split(x)                # (B, O, N)
        sigma = log_sigma.exp()                        # (B, O, N)
        eps = torch.randn(*mu.shape, S, device=x.device)  # (B, O, N, S)
        return mu.unsqueeze(-1) + sigma.unsqueeze(-1) * eps  # (B, O, N, S)
