from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import torch

from ..model.MultiscaleConvForecaster import MultiscaleConvForecaster
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)

_DEFAULT_KERNELS = (3, 7, 15, 31)


@dataclass
class MultiscaleConvForecasterParameters:
    d_model: int = 64
    n_layers: int = 3
    kernels: Tuple[int, ...] = _DEFAULT_KERNELS
    dropout: float = 0.1
    revin: bool = True


@dataclass
class MultiscaleConvForecasterForecast(ForecastExp, MultiscaleConvForecasterParameters):
    model_type: str = "MultiscaleConvForecaster"

    def _init_model(self):
        self.model = MultiscaleConvForecaster(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            n_layers=self.n_layers,
            kernels=self.kernels,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class MultiscaleConvForecasterUEAClassification(UEAClassificationExp, MultiscaleConvForecasterParameters):
    model_type: str = "MultiscaleConvForecaster"

    def _init_model(self):
        self.model = MultiscaleConvForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            n_layers=self.n_layers,
            kernels=self.kernels,
            dropout=self.dropout,
            revin=False,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class MultiscaleConvForecasterAnomalyDetection(AnomalyDetectionExp, MultiscaleConvForecasterParameters):
    model_type: str = "MultiscaleConvForecaster"

    def _init_model(self):
        self.model = MultiscaleConvForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            n_layers=self.n_layers,
            kernels=self.kernels,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class MultiscaleConvForecasterImputation(ImputationExp, MultiscaleConvForecasterParameters):
    model_type: str = "MultiscaleConvForecaster"

    def _init_model(self):
        self.model = MultiscaleConvForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            n_layers=self.n_layers,
            kernels=self.kernels,
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
