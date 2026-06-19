"""Deep Ensemble probabilistic forecaster.

Trains N independent backbone copies with different random initialisations.
At inference each member produces a point prediction; the N predictions form
a distributional ensemble (B, pred_len, enc_in, N).

Reference: Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty
Estimation using Deep Ensembles", NeurIPS 2017.
https://arxiv.org/abs/1612.01474
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleForecaster(nn.Module):
    """Deep Ensemble that wraps N independently initialised backbones.

    All members are trained simultaneously in the same forward pass.
    The training objective is the mean MSE loss across all members
    (identical to training each member independently on the same batch).

    Args:
        backbone_fn: Zero-argument callable that returns a fresh ``nn.Module``
            with signature ``forward(x) -> (B, pred_len, enc_in)``.  Called
            ``num_members`` times; each call should produce a *new* module with
            independently randomised weights.
        num_members: Number of ensemble members.
    """

    def __init__(self, backbone_fn: Callable[[], nn.Module], num_members: int = 5):
        super().__init__()
        if num_members < 1:
            raise ValueError("num_members must be >= 1")
        self.num_members = num_members
        self.members = nn.ModuleList([backbone_fn() for _ in range(num_members)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the ensemble mean — used as the point forecast during training.

        Args:
            x: input ``(B, seq_len, enc_in)``.

        Returns:
            Mean prediction ``(B, pred_len, enc_in)``.
        """
        preds = torch.stack([m(x) for m in self.members], dim=-1)  # (B, O, N, M)
        return preds.mean(dim=-1)

    def mse_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mean MSE loss across all ensemble members.

        Args:
            x: input ``(B, seq_len, enc_in)``.
            y: target ``(B, pred_len, enc_in)``.

        Returns:
            Scalar mean MSE.
        """
        preds = torch.stack([m(x) for m in self.members], dim=-1)  # (B, O, N, M)
        y_exp = y.unsqueeze(-1).expand_as(preds)
        return F.mse_loss(preds, y_exp)

    @torch.no_grad()
    def sample(
        self, x: torch.Tensor, num_samples: Optional[int] = None
    ) -> torch.Tensor:
        """Return ensemble predictions as a distributional sample set.

        Args:
            x: input ``(B, seq_len, enc_in)``.
            num_samples: if given, select the first ``num_samples`` members
                (must be <= ``num_members``).

        Returns:
            ``(B, pred_len, enc_in, M)`` stack of member predictions.
        """
        M = num_samples if num_samples is not None else self.num_members
        M = min(M, self.num_members)
        preds = torch.stack([self.members[i](x) for i in range(M)], dim=-1)
        return preds  # (B, O, N, M)
