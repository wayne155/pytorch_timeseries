from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.HyperForecaster import HyperForecaster
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class HyperForecasterParameters:
    d_ctx: int = 64
    hidden: int = 32
    d_ctx_hidden: int = 128
    dropout: float = 0.1
    revin: bool = True


@dataclass
class HyperForecasterForecast(ForecastExp, HyperForecasterParameters):
    model_type: str = "HyperForecaster"

    def _init_model(self):
        self.model = HyperForecaster(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            d_ctx=self.d_ctx,
            hidden=self.hidden,
            d_ctx_hidden=self.d_ctx_hidden,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class HyperForecasterUEAClassification(UEAClassificationExp, HyperForecasterParameters):
    model_type: str = "HyperForecaster"

    def _init_model(self):
        self.model = HyperForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_ctx=self.d_ctx,
            hidden=self.hidden,
            d_ctx_hidden=self.d_ctx_hidden,
            dropout=self.dropout,
            revin=False,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class HyperForecasterAnomalyDetection(AnomalyDetectionExp, HyperForecasterParameters):
    model_type: str = "HyperForecaster"

    def _init_model(self):
        self.model = HyperForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_ctx=self.d_ctx,
            hidden=self.hidden,
            d_ctx_hidden=self.d_ctx_hidden,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class HyperForecasterImputation(ImputationExp, HyperForecasterParameters):
    model_type: str = "HyperForecaster"

    def _init_model(self):
        self.model = HyperForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_ctx=self.d_ctx,
            hidden=self.hidden,
            d_ctx_hidden=self.d_ctx_hidden,
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
