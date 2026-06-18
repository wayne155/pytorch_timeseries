"""MC-Dropout wrapper for probabilistic forecasting (Gal & Ghahramani, 2016).

Training is identical to point forecasting: one stochastic forward pass with
MSE/MAE loss.  Uncertainty is obtained at inference time by sampling *S*
stochastic forward passes with dropout left active (backbone stays in train
mode so its Dropout layers fire).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MCDropoutForecaster(nn.Module):
    """Wrap a dropout-bearing backbone for MC-Dropout probabilistic forecasting.

    Args:
        backbone: any ``nn.Module`` whose ``forward(x)`` accepts
            ``(B, seq_len, enc_in)`` and returns ``(B, pred_len, enc_in)``.
        num_samples: default ensemble size *S* used by :meth:`sample`.
    """

    def __init__(self, backbone: nn.Module, num_samples: int = 50):
        super().__init__()
        self.backbone = backbone
        self.num_samples = num_samples

    # ------------------------------------------------------------------ #
    # point forward (training)                                            #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single-pass forward used during training.

        Returns:
            Tensor ``(B, pred_len, enc_in)`` — identical to backbone output.
        """
        return self.backbone(x)

    # ------------------------------------------------------------------ #
    # ensemble sampling (inference)                                       #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def sample(
        self, x: torch.Tensor, num_samples: Optional[int] = None
    ) -> torch.Tensor:
        """Draw an ensemble with stochastic dropout.

        The backbone is kept in *train* mode (so Dropout layers fire) while
        ``torch.no_grad()`` avoids unnecessary gradient bookkeeping.

        Args:
            x: input batch, shape ``(B, seq_len, enc_in)``.
            num_samples: override for ``self.num_samples``.

        Returns:
            Tensor ``(B, pred_len, enc_in, S)`` where *S* is the sample count.
        """
        S = num_samples if num_samples is not None else self.num_samples
        was_training = self.backbone.training
        self.backbone.train()
        samples = torch.stack(
            [self.backbone(x) for _ in range(S)], dim=-1
        )  # (B, pred_len, enc_in, S)
        if not was_training:
            self.backbone.eval()
        return samples
