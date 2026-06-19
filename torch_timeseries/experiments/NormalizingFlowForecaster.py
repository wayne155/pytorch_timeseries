"""Normalizing Flow probabilistic forecasting experiment.

A Real-NVP affine coupling flow conditioned on a VanillaTransformer backbone
provides an exact-likelihood distribution over the forecast horizon.
Training: minimise NLL = -log p_flow(y|x).
Inference: sample from the flow posterior via inverse flow + N(0,I).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..dataloader.v2 import TSBatch
from ..model.NormalizingFlowForecaster import NormalizingFlowForecaster
from .prob_forecast import ProbForecastExp


@dataclass
class NormalizingFlowParameters:
    d_model: int = 256
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    revin: bool = True
    flow_layers: int = 6
    flow_hidden: int = 128


@dataclass
class NormalizingFlowForecast(NormalizingFlowParameters, ProbForecastExp):
    """Exact-likelihood normalizing flow probabilistic forecasting."""

    model_type: str = "NormalizingFlow"

    def _build_model(self) -> NormalizingFlowForecaster:
        return NormalizingFlowForecaster(
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
            flow_layers=self.flow_layers,
            flow_hidden=self.flow_hidden,
            num_samples=self.num_samples,
        )

    def _process_train_batch(self, batch: TSBatch) -> torch.Tensor:
        x = batch.x.to(self.device)
        y = batch.y.to(self.device)
        return self.model.nll_loss(x, y)

    def _process_val_batch(self, batch: TSBatch):
        x = batch.x.to(self.device)
        y = batch.y.to(self.device)
        preds = self.model.sample(x)   # (B, pred_len, N, S)
        return preds, y
