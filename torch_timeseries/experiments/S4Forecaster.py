from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.S4Forecaster import S4Forecaster
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class S4ForecasterParameters:
    d_model: int = 64
    d_state: int = 32
    n_layers: int = 3
    mlp_mult: int = 2
    dropout: float = 0.1
    revin: bool = True


@dataclass
class S4ForecasterForecast(ForecastExp, S4ForecasterParameters):
    model_type: str = "S4Forecaster"

    def _init_model(self):
        self.model = S4Forecaster(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            d_state=self.d_state,
            n_layers=self.n_layers,
            mlp_mult=self.mlp_mult,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class S4ForecasterUEAClassification(UEAClassificationExp, S4ForecasterParameters):
    model_type: str = "S4Forecaster"

    def _init_model(self):
        self.model = S4Forecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            d_state=self.d_state,
            n_layers=self.n_layers,
            mlp_mult=self.mlp_mult,
            dropout=self.dropout,
            revin=False,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class S4ForecasterAnomalyDetection(AnomalyDetectionExp, S4ForecasterParameters):
    model_type: str = "S4Forecaster"

    def _init_model(self):
        self.model = S4Forecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            d_state=self.d_state,
            n_layers=self.n_layers,
            mlp_mult=self.mlp_mult,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class S4ForecasterImputation(ImputationExp, S4ForecasterParameters):
    model_type: str = "S4Forecaster"

    def _init_model(self):
        self.model = S4Forecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            d_state=self.d_state,
            n_layers=self.n_layers,
            mlp_mult=self.mlp_mult,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(
        self,
        batch_masked_x,
        batch_x,
        batch_origin_x,
        batch_mask,
        batch_x_date_enc,
    ):
        batch_masked_x = batch_masked_x.to(self.device, dtype=torch.float32)
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_masked_x)
        return outputs, batch_x
