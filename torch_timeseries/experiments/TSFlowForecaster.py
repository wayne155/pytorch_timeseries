"""TSFLow probabilistic forecasting experiment.

Conditional Flow Matching with a Gaussian Process prior (Kollovieh et al.).
Training: minimise the CFM loss (MSE between predicted and target velocity).
Inference: sample S trajectories via Euler ODE from GP noise to forecast.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..dataloader.v2 import TSBatch
from ..model.TSFlowForecaster import TSFlowForecaster
from .prob_forecast import ProbForecastExp


@dataclass
class TSFlowParameters:
    d_model: int = 128
    flow_layers: int = 3
    gp_length_scale: float = 2.0
    n_steps: int = 20


@dataclass
class TSFlowForecast(TSFlowParameters, ProbForecastExp):
    """Flow Matching with GP Priors probabilistic forecasting."""

    model_type: str = "TSFlow"

    def _build_model(self) -> TSFlowForecaster:
        return TSFlowForecaster(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            flow_layers=self.flow_layers,
            gp_length_scale=self.gp_length_scale,
            n_steps=self.n_steps,
            num_samples=self.num_samples,
        )

    def _process_train_batch(self, batch: TSBatch) -> torch.Tensor:
        x = batch.x.to(self.device)
        y = batch.y.to(self.device)
        return self.model.flow_loss(x, y)

    def _process_val_batch(self, batch: TSBatch):
        x = batch.x.to(self.device)
        y = batch.y.to(self.device)
        preds = self.model.sample(x)    # (B, pred_len, N, S)
        return preds, y
