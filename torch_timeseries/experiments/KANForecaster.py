from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.KANForecaster import KANForecaster
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class KANForecasterParameters:
    hidden: int = 64
    e_layers: int = 2
    degree: int = 5
    dropout: float = 0.1
    revin: bool = True


@dataclass
class KANForecasterForecast(ForecastExp, KANForecasterParameters):
    model_type: str = "KANForecaster"

    def _init_model(self):
        self.model = KANForecaster(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            hidden=self.hidden,
            e_layers=self.e_layers,
            degree=self.degree,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class KANForecasterUEAClassification(UEAClassificationExp, KANForecasterParameters):
    model_type: str = "KANForecaster"

    def _init_model(self):
        self.model = KANForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            hidden=self.hidden,
            e_layers=self.e_layers,
            degree=self.degree,
            dropout=self.dropout,
            revin=False,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class KANForecasterAnomalyDetection(AnomalyDetectionExp, KANForecasterParameters):
    model_type: str = "KANForecaster"

    def _init_model(self):
        self.model = KANForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            hidden=self.hidden,
            e_layers=self.e_layers,
            degree=self.degree,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class KANForecasterImputation(ImputationExp, KANForecasterParameters):
    model_type: str = "KANForecaster"

    def _init_model(self):
        self.model = KANForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            hidden=self.hidden,
            e_layers=self.e_layers,
            degree=self.degree,
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
