from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.FourierMixerForecaster import FourierMixerForecaster
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class FourierMixerForecasterParameters:
    e_layers: int = 3
    dropout: float = 0.1
    revin: bool = True


@dataclass
class FourierMixerForecasterForecast(ForecastExp, FourierMixerForecasterParameters):
    model_type: str = "FourierMixerForecaster"

    def _init_model(self):
        self.model = FourierMixerForecaster(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            e_layers=self.e_layers,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class FourierMixerForecasterUEAClassification(UEAClassificationExp, FourierMixerForecasterParameters):
    model_type: str = "FourierMixerForecaster"

    def _init_model(self):
        self.model = FourierMixerForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            e_layers=self.e_layers,
            dropout=self.dropout,
            revin=False,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class FourierMixerForecasterAnomalyDetection(AnomalyDetectionExp, FourierMixerForecasterParameters):
    model_type: str = "FourierMixerForecaster"

    def _init_model(self):
        self.model = FourierMixerForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            e_layers=self.e_layers,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class FourierMixerForecasterImputation(ImputationExp, FourierMixerForecasterParameters):
    model_type: str = "FourierMixerForecaster"

    def _init_model(self):
        self.model = FourierMixerForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            e_layers=self.e_layers,
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
