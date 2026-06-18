from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..model.SparseTSF import SparseTSF
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class SparseTSFParameters:
    period: Optional[int] = None   # None → seq_len // 4 (quarter-frequency)
    revin: bool = True


@dataclass
class SparseTSFForecast(ForecastExp, SparseTSFParameters):
    model_type: str = "SparseTSF"

    def _init_model(self):
        self.model = SparseTSF(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            period=self.period,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class SparseTSFUEAClassification(UEAClassificationExp, SparseTSFParameters):
    model_type: str = "SparseTSF"

    def _init_model(self):
        self.model = SparseTSF(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            period=self.period,
            revin=False,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class SparseTSFAnomalyDetection(AnomalyDetectionExp, SparseTSFParameters):
    model_type: str = "SparseTSF"

    def _init_model(self):
        self.model = SparseTSF(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            period=self.period,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class SparseTSFImputation(ImputationExp, SparseTSFParameters):
    model_type: str = "SparseTSF"

    def _init_model(self):
        self.model = SparseTSF(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            period=self.period,
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
