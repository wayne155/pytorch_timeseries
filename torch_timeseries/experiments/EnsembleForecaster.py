"""Deep Ensemble probabilistic forecasting experiment."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..dataloader.v2 import TSBatch
from ..model.EnsembleForecaster import EnsembleForecaster
from ..model.VanillaTransformer import VanillaTransformer
from .prob_forecast import ProbForecastExp


@dataclass
class EnsembleParameters:
    d_model: int = 256
    n_heads: int = 4
    e_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    revin: bool = True
    num_members: int = 5


@dataclass
class EnsembleForecast(EnsembleParameters, ProbForecastExp):
    """Deep Ensemble probabilistic forecasting (MSE training, ensemble inference)."""

    model_type: str = "Ensemble"

    def _build_model(self) -> EnsembleForecaster:
        enc_in = self.dataset.num_features

        def backbone_fn():
            return VanillaTransformer(
                seq_len=self.windows,
                pred_len=self.pred_len,
                enc_in=enc_in,
                d_model=self.d_model,
                n_heads=self.n_heads,
                e_layers=self.e_layers,
                d_ff=self.d_ff,
                dropout=self.dropout,
                activation=self.activation,
                revin=self.revin,
            )

        return EnsembleForecaster(
            backbone_fn=backbone_fn,
            num_members=self.num_members,
        )

    def _process_train_batch(self, batch: TSBatch) -> torch.Tensor:
        x = batch.x.to(self.device)
        y = batch.y.to(self.device)
        return self.model.mse_loss(x, y)

    def _process_val_batch(self, batch: TSBatch):
        x = batch.x.to(self.device)
        y = batch.y.to(self.device)
        preds = self.model.sample(x)   # (B, pred_len, N, M)
        return preds, y
