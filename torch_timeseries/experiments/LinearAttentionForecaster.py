from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.LinearAttentionForecaster import LinearAttentionForecaster
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class LinearAttentionForecasterParameters:
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 128
    patch_len: int = 16
    stride: int = 16
    dropout: float = 0.1
    revin: bool = True


@dataclass
class LinearAttentionForecasterForecast(ForecastExp, LinearAttentionForecasterParameters):
    model_type: str = "LinearAttentionForecaster"

    def _init_model(self):
        self.model = LinearAttentionForecaster(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            patch_len=self.patch_len,
            stride=self.stride,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class LinearAttentionForecasterUEAClassification(UEAClassificationExp, LinearAttentionForecasterParameters):
    model_type: str = "LinearAttentionForecaster"

    def _init_model(self):
        self.model = LinearAttentionForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            patch_len=self.patch_len,
            stride=self.stride,
            dropout=self.dropout,
            revin=False,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class LinearAttentionForecasterAnomalyDetection(AnomalyDetectionExp, LinearAttentionForecasterParameters):
    model_type: str = "LinearAttentionForecaster"

    def _init_model(self):
        self.model = LinearAttentionForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            patch_len=self.patch_len,
            stride=self.stride,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class LinearAttentionForecasterImputation(ImputationExp, LinearAttentionForecasterParameters):
    model_type: str = "LinearAttentionForecaster"

    def _init_model(self):
        self.model = LinearAttentionForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            patch_len=self.patch_len,
            stride=self.stride,
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
