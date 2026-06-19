from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.MICN import MICN
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class MICNParameters:
    d_model: int = 64
    num_scales: int = 3
    kernel_size: int = 5
    dropout: float = 0.05
    revin: bool = True


@dataclass
class MICNForecast(ForecastExp, MICNParameters):
    model_type: str = "MICN"

    def _init_model(self):
        self.model = MICN(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            num_scales=self.num_scales,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class MICNUEAClassification(UEAClassificationExp, MICNParameters):
    model_type: str = "MICN"

    def _init_model(self):
        self.model = MICN(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            num_scales=self.num_scales,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            revin=False,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class MICNAnomalyDetection(AnomalyDetectionExp, MICNParameters):
    model_type: str = "MICN"

    def _init_model(self):
        self.model = MICN(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            num_scales=self.num_scales,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class MICNImputation(ImputationExp, MICNParameters):
    model_type: str = "MICN"

    def _init_model(self):
        self.model = MICN(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            num_scales=self.num_scales,
            kernel_size=self.kernel_size,
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
