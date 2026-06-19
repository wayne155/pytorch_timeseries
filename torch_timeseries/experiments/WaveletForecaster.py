from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.WaveletForecaster import WaveletForecaster
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class WaveletForecasterParameters:
    n_levels: int = 3
    revin: bool = True


@dataclass
class WaveletForecasterForecast(ForecastExp, WaveletForecasterParameters):
    model_type: str = "WaveletForecaster"

    def _init_model(self):
        self.model = WaveletForecaster(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            n_levels=self.n_levels,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class WaveletForecasterUEAClassification(UEAClassificationExp, WaveletForecasterParameters):
    model_type: str = "WaveletForecaster"

    def _init_model(self):
        self.model = WaveletForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            n_levels=self.n_levels,
            revin=False,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class WaveletForecasterAnomalyDetection(AnomalyDetectionExp, WaveletForecasterParameters):
    model_type: str = "WaveletForecaster"

    def _init_model(self):
        self.model = WaveletForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            n_levels=self.n_levels,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class WaveletForecasterImputation(ImputationExp, WaveletForecasterParameters):
    model_type: str = "WaveletForecaster"

    def _init_model(self):
        self.model = WaveletForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            n_levels=self.n_levels,
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
