from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.RLinear import RLinear
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class RLinearParameters:
    individual: bool = True


@dataclass
class RLinearForecast(ForecastExp, RLinearParameters):
    model_type: str = "RLinear"

    def _init_model(self):
        self.model = RLinear(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            individual=self.individual,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class RLinearUEAClassification(UEAClassificationExp, RLinearParameters):
    model_type: str = "RLinear"

    def _init_model(self):
        self.model = RLinear(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            individual=self.individual,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class RLinearAnomalyDetection(AnomalyDetectionExp, RLinearParameters):
    model_type: str = "RLinear"

    def _init_model(self):
        self.model = RLinear(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            individual=self.individual,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class RLinearImputation(ImputationExp, RLinearParameters):
    model_type: str = "RLinear"

    def _init_model(self):
        self.model = RLinear(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            individual=self.individual,
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
