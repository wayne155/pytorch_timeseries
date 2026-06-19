from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.CycleNet import CycleNet
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class CycleNetParameters:
    cycle_len: int = 24
    backbone: str = "linear"
    d_model: int = 512
    revin: bool = True
    dropout: float = 0.0


@dataclass
class CycleNetForecast(ForecastExp, CycleNetParameters):
    model_type: str = "CycleNet"

    def _init_model(self):
        self.model = CycleNet(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            cycle_len=self.cycle_len,
            backbone=self.backbone,
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
class CycleNetUEAClassification(UEAClassificationExp, CycleNetParameters):
    model_type: str = "CycleNet"

    def _init_model(self):
        self.model = CycleNet(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            cycle_len=self.cycle_len,
            backbone=self.backbone,
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
class CycleNetAnomalyDetection(AnomalyDetectionExp, CycleNetParameters):
    model_type: str = "CycleNet"

    def _init_model(self):
        self.model = CycleNet(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            cycle_len=self.cycle_len,
            backbone=self.backbone,
            d_model=self.d_model,
            revin=self.revin,
            dropout=self.dropout,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class CycleNetImputation(ImputationExp, CycleNetParameters):
    model_type: str = "CycleNet"

    def _init_model(self):
        self.model = CycleNet(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            cycle_len=self.cycle_len,
            backbone=self.backbone,
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
