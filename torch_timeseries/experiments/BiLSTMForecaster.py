from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.BiLSTMForecaster import BiLSTMForecaster
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class BiLSTMForecasterParameters:
    d_model: int = 64
    num_layers: int = 2
    d_attn: int = 32
    dropout: float = 0.1
    revin: bool = True


@dataclass
class BiLSTMForecasterForecast(ForecastExp, BiLSTMForecasterParameters):
    model_type: str = "BiLSTMForecaster"

    def _init_model(self):
        self.model = BiLSTMForecaster(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            num_layers=self.num_layers,
            d_attn=self.d_attn,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class BiLSTMForecasterUEAClassification(UEAClassificationExp, BiLSTMForecasterParameters):
    model_type: str = "BiLSTMForecaster"

    def _init_model(self):
        self.model = BiLSTMForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            num_layers=self.num_layers,
            d_attn=self.d_attn,
            dropout=self.dropout,
            revin=False,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class BiLSTMForecasterAnomalyDetection(AnomalyDetectionExp, BiLSTMForecasterParameters):
    model_type: str = "BiLSTMForecaster"

    def _init_model(self):
        self.model = BiLSTMForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            num_layers=self.num_layers,
            d_attn=self.d_attn,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class BiLSTMForecasterImputation(ImputationExp, BiLSTMForecasterParameters):
    model_type: str = "BiLSTMForecaster"

    def _init_model(self):
        self.model = BiLSTMForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            num_layers=self.num_layers,
            d_attn=self.d_attn,
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
