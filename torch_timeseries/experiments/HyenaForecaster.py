from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.HyenaForecaster import HyenaForecaster
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class HyenaForecasterParameters:
    d_model: int = 64
    n_layers: int = 3
    pos_freqs: int = 16
    filter_dim: int = 64
    dropout: float = 0.1
    revin: bool = True


@dataclass
class HyenaForecasterForecast(ForecastExp, HyenaForecasterParameters):
    model_type: str = "HyenaForecaster"

    def _init_model(self):
        self.model = HyenaForecaster(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            n_layers=self.n_layers,
            pos_freqs=self.pos_freqs,
            filter_dim=self.filter_dim,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class HyenaForecasterUEAClassification(UEAClassificationExp, HyenaForecasterParameters):
    model_type: str = "HyenaForecaster"

    def _init_model(self):
        self.model = HyenaForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            n_layers=self.n_layers,
            pos_freqs=self.pos_freqs,
            filter_dim=self.filter_dim,
            dropout=self.dropout,
            revin=False,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class HyenaForecasterAnomalyDetection(AnomalyDetectionExp, HyenaForecasterParameters):
    model_type: str = "HyenaForecaster"

    def _init_model(self):
        self.model = HyenaForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            n_layers=self.n_layers,
            pos_freqs=self.pos_freqs,
            filter_dim=self.filter_dim,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class HyenaForecasterImputation(ImputationExp, HyenaForecasterParameters):
    model_type: str = "HyenaForecaster"

    def _init_model(self):
        self.model = HyenaForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            n_layers=self.n_layers,
            pos_freqs=self.pos_freqs,
            filter_dim=self.filter_dim,
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
