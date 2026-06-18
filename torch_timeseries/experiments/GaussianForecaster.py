"""Gaussian heteroscedastic probabilistic forecasting experiment.

The backbone (VanillaTransformer by default) is trained to minimise the
Gaussian NLL.  At inference, samples are drawn from N(μ, σ²) per window.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..dataloader.v2 import TSBatch
from ..model.GaussianForecaster import GaussianForecaster
from ..model.VanillaTransformer import VanillaTransformer
from .prob_forecast import ProbForecastExp


@dataclass
class GaussianParameters:
    """Backbone hyper-parameters shared by all Gaussian experiment wrappers."""

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
class GaussianForecast(GaussianParameters, ProbForecastExp):
    """Heteroscedastic Gaussian probabilistic forecasting (NLL training)."""

    model_type: str = "Gaussian"

    def _build_model(self) -> GaussianForecaster:
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
        return GaussianForecaster(
            backbone=backbone,
            enc_in=self.dataset.num_features,
            pred_len=self.pred_len,
            num_samples=self.num_samples,
        )

    def _process_train_batch(self, batch: TSBatch) -> torch.Tensor:
        x = batch.x.to(self.device)
        y = batch.y.to(self.device)
        return self.model.nll_loss(x, y)

    def _process_val_batch(self, batch: TSBatch):
        x = batch.x.to(self.device)
        y = batch.y.to(self.device)
        preds = self.model.sample(x)  # (B, pred_len, N, S)
        return preds, y
