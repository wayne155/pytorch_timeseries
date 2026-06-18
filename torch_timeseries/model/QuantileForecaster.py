"""Quantile regression distributional forecasting head.

Trains a set of K quantile predictors jointly using the pinball (quantile) loss.
At inference the K quantile outputs serve as a distributional ensemble.

Reference: Koenker & Basset (1978) regression quantiles; Taylor (2000) quantile
regression for interval forecasting in time series.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


_DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class QuantileForecaster(nn.Module):
    """Attach K quantile heads to a point-forecasting backbone.

    The backbone produces ``(B, pred_len, enc_in)`` outputs.  A single
    linear quantile head maps the flattened backbone output to K quantile
    predictions simultaneously.

    At training time ``pinball_loss`` optimises all quantiles jointly.
    At inference time ``sample()`` returns the sorted quantile predictions
    as a ``(B, pred_len, enc_in, K)`` tensor — compatible with the
    probabilistic metrics API.

    Args:
        backbone: ``nn.Module`` with signature ``forward(x) -> (B, O, N)``.
        enc_in: number of output channels.
        pred_len: forecasting horizon.
        quantiles: quantile levels to predict (must be strictly increasing,
            values in (0, 1)).
    """

    def __init__(
        self,
        backbone: nn.Module,
        enc_in: int,
        pred_len: int,
        quantiles: Optional[List[float]] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.enc_in = enc_in
        self.pred_len = pred_len
        self.quantiles: List[float] = quantiles if quantiles is not None else list(_DEFAULT_QUANTILES)

        if len(self.quantiles) < 1:
            raise ValueError("quantiles must be non-empty")
        if sorted(self.quantiles) != self.quantiles:
            raise ValueError("quantiles must be strictly increasing")
        if any(q <= 0 or q >= 1 for q in self.quantiles):
            raise ValueError("all quantiles must be in (0, 1)")

        K = len(self.quantiles)
        flat_dim = enc_in * pred_len
        self.quantile_head = nn.Linear(flat_dim, flat_dim * K)
        nn.init.zeros_(self.quantile_head.weight)
        nn.init.zeros_(self.quantile_head.bias)

    @property
    def num_quantiles(self) -> int:
        return len(self.quantiles)

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return quantile predictions ``(B, O, N, K)``."""
        feat = self.backbone(x)          # (B, O, N)
        B, O, N = feat.shape
        K = self.num_quantiles
        q_preds = self.quantile_head(feat.flatten(1)).view(B, O, N, K)
        return q_preds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return quantile predictions ``(B, O, N, K)`` used during training."""
        return self._predict(x)

    def pinball_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Sum of pinball (quantile) losses over all K quantiles.

        Args:
            x: input ``(B, seq_len, enc_in)``.
            y: target ``(B, pred_len, enc_in)``.

        Returns:
            Scalar mean pinball loss.
        """
        q_preds = self._predict(x)                             # (B, O, N, K)
        y_exp = y.unsqueeze(-1)                                # (B, O, N, 1)
        q_tensor = x.new_tensor(self.quantiles)                # (K,)
        errors = y_exp - q_preds                               # (B, O, N, K)
        loss = torch.where(
            errors >= 0,
            q_tensor * errors,
            (q_tensor - 1.0) * errors,
        )
        return loss.mean()

    @torch.no_grad()
    def sample(
        self, x: torch.Tensor, num_samples: Optional[int] = None
    ) -> torch.Tensor:
        """Return sorted quantile predictions as a distributional ensemble.

        Args:
            x: input ``(B, seq_len, enc_in)``.
            num_samples: unused (kept for API consistency with other forecasters).

        Returns:
            ``(B, pred_len, enc_in, K)`` tensor of sorted quantile predictions.
        """
        q_preds = self._predict(x)          # (B, O, N, K)
        q_preds, _ = q_preds.sort(dim=-1)  # enforce monotonicity at inference
        return q_preds
