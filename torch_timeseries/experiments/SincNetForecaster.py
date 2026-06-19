from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.SincNetForecaster import SincNetForecaster
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class SincNetForecasterParameters:
    n_filters: int = 32
    kernel_size: int = 25
    n_conv_layers: int = 2
    dropout: float = 0.1
    revin: bool = True


@dataclass
class SincNetForecasterForecast(ForecastExp, SincNetForecasterParameters):
    model_type: str = "SincNetForecaster"

    def _init_model(self):
        self.model = SincNetForecaster(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            n_conv_layers=self.n_conv_layers,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class SincNetForecasterUEAClassification(UEAClassificationExp, SincNetForecasterParameters):
    model_type: str = "SincNetForecaster"

    def _init_model(self):
        self.model = SincNetForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            n_conv_layers=self.n_conv_layers,
            dropout=self.dropout,
            revin=False,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class SincNetForecasterAnomalyDetection(AnomalyDetectionExp, SincNetForecasterParameters):
    model_type: str = "SincNetForecaster"

    def _init_model(self):
        self.model = SincNetForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            n_conv_layers=self.n_conv_layers,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class SincNetForecasterImputation(ImputationExp, SincNetForecasterParameters):
    model_type: str = "SincNetForecaster"

    def _init_model(self):
        self.model = SincNetForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            n_conv_layers=self.n_conv_layers,
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
