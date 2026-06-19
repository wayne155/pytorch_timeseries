from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.WaveNet import WaveNet
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class WaveNetParameters:
    d_model: int = 64
    d_skip: int = 64
    kernel_size: int = 2
    num_layers: int = 8
    num_stacks: int = 1
    dropout: float = 0.0
    revin: bool = True


@dataclass
class WaveNetForecast(ForecastExp, WaveNetParameters):
    model_type: str = "WaveNet"

    def _init_model(self):
        self.model = WaveNet(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            d_skip=self.d_skip,
            kernel_size=self.kernel_size,
            num_layers=self.num_layers,
            num_stacks=self.num_stacks,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class WaveNetUEAClassification(UEAClassificationExp, WaveNetParameters):
    model_type: str = "WaveNet"

    def _init_model(self):
        self.model = WaveNet(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            d_skip=self.d_skip,
            kernel_size=self.kernel_size,
            num_layers=self.num_layers,
            num_stacks=self.num_stacks,
            dropout=self.dropout,
            revin=False,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class WaveNetAnomalyDetection(AnomalyDetectionExp, WaveNetParameters):
    model_type: str = "WaveNet"

    def _init_model(self):
        self.model = WaveNet(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            d_skip=self.d_skip,
            kernel_size=self.kernel_size,
            num_layers=self.num_layers,
            num_stacks=self.num_stacks,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class WaveNetImputation(ImputationExp, WaveNetParameters):
    model_type: str = "WaveNet"

    def _init_model(self):
        self.model = WaveNet(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            d_skip=self.d_skip,
            kernel_size=self.kernel_size,
            num_layers=self.num_layers,
            num_stacks=self.num_stacks,
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
