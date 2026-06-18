"""MC-Dropout probabilistic forecasting experiment.

Wraps a VanillaTransformer backbone in MCDropoutForecaster and registers it
under ``model_type = "MCDropout"`` for all four standard tasks.  Training is
point-regression (MSE); uncertainty is obtained at inference via stochastic
dropout sampling.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..dataloader.v2 import TSBatch
from ..model.MCDropoutForecaster import MCDropoutForecaster
from ..model.VanillaTransformer import VanillaTransformer
from .prob_forecast import ProbForecastExp


@dataclass
class MCDropoutParameters:
    """Hyper-parameters shared across all MCDropout experiment wrappers."""

    d_model: int = 256
    n_heads: int = 4
    e_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    revin: bool = True


# ------------------------------------------------------------------ #
# Forecast                                                            #
# ------------------------------------------------------------------ #


@dataclass
class MCDropoutForecast(MCDropoutParameters, ProbForecastExp):
    """Probabilistic long-term forecasting with MC-Dropout."""

    model_type: str = "MCDropout"

    def _build_model(self) -> MCDropoutForecaster:
        backbone = VanillaTransformer(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            revin=self.revin,
        )
        return MCDropoutForecaster(backbone, num_samples=self.num_samples)

    def _process_train_batch(self, batch: TSBatch) -> torch.Tensor:
        x = batch.x.to(self.device)
        y = batch.y.to(self.device)
        pred = self.model(x)  # point forward pass
        return F.mse_loss(pred, y)

    def _process_val_batch(self, batch: TSBatch):
        x = batch.x.to(self.device)
        y = batch.y.to(self.device)
        preds = self.model.sample(x)  # (B, pred_len, N, S)
        return preds, y
