"""Student-T heteroscedastic probabilistic forecasting experiment."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..dataloader.v2 import TSBatch
from ..model.StudentTForecaster import StudentTForecaster
from ..model.VanillaTransformer import VanillaTransformer
from .prob_forecast import ProbForecastExp


@dataclass
class StudentTParameters:
    d_model: int = 256
    n_heads: int = 4
    e_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    revin: bool = True


@dataclass
class StudentTForecast(StudentTParameters, ProbForecastExp):
    """Heteroscedastic Student-T probabilistic forecasting (NLL training)."""

    model_type: str = "StudentT"

    def _build_model(self) -> StudentTForecaster:
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
        return StudentTForecaster(
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
