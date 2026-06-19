from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..model.LightTS import LightTS
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class LightTSParameters:
    chunk_size: Optional[int] = None
    d_model: int = 64
    revin: bool = True
    dropout: float = 0.0


@dataclass
class LightTSForecast(ForecastExp, LightTSParameters):
    model_type: str = "LightTS"

    def _init_model(self):
        self.model = LightTS(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            chunk_size=self.chunk_size,
            d_model=self.d_model,
            revin=self.revin,
            dropout=self.dropout,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class LightTSUEAClassification(UEAClassificationExp, LightTSParameters):
    model_type: str = "LightTS"

    def _init_model(self):
        self.model = LightTS(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            chunk_size=self.chunk_size,
            d_model=self.d_model,
            revin=False,
            dropout=self.dropout,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class LightTSAnomalyDetection(AnomalyDetectionExp, LightTSParameters):
    model_type: str = "LightTS"

    def _init_model(self):
        self.model = LightTS(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            chunk_size=self.chunk_size,
            d_model=self.d_model,
            revin=self.revin,
            dropout=self.dropout,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class LightTSImputation(ImputationExp, LightTSParameters):
    model_type: str = "LightTS"

    def _init_model(self):
        self.model = LightTS(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            chunk_size=self.chunk_size,
            d_model=self.d_model,
            revin=self.revin,
            dropout=self.dropout,
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
