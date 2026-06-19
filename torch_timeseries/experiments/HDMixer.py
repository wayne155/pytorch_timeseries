from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch

from ..model.HDMixer import HDMixer
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class HDMixerParameters:
    patch_sizes: List[int] = field(default_factory=lambda: [4, 8, 16])
    d_model: int = 64
    dropout: float = 0.1
    revin: bool = True


@dataclass
class HDMixerForecast(ForecastExp, HDMixerParameters):
    model_type: str = "HDMixer"

    def _init_model(self):
        self.model = HDMixer(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            patch_sizes=self.patch_sizes,
            d_model=self.d_model,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class HDMixerUEAClassification(UEAClassificationExp, HDMixerParameters):
    model_type: str = "HDMixer"

    def _init_model(self):
        self.model = HDMixer(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            patch_sizes=self.patch_sizes,
            d_model=self.d_model,
            dropout=self.dropout,
            revin=False,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class HDMixerAnomalyDetection(AnomalyDetectionExp, HDMixerParameters):
    model_type: str = "HDMixer"

    def _init_model(self):
        self.model = HDMixer(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            patch_sizes=self.patch_sizes,
            d_model=self.d_model,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class HDMixerImputation(ImputationExp, HDMixerParameters):
    model_type: str = "HDMixer"

    def _init_model(self):
        self.model = HDMixer(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            patch_sizes=self.patch_sizes,
            d_model=self.d_model,
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
