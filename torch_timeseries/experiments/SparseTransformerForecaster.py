from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.SparseTransformerForecaster import SparseTransformerForecaster
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class SparseTransformerForecasterParameters:
    patch_size: int = 8
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 128
    local_window: int = 3
    stride: int = 4
    dropout: float = 0.1
    revin: bool = True


@dataclass
class SparseTransformerForecasterForecast(ForecastExp, SparseTransformerForecasterParameters):
    model_type: str = "SparseTransformerForecaster"

    def _init_model(self):
        self.model = SparseTransformerForecaster(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            patch_size=self.patch_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            local_window=self.local_window,
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
class SparseTransformerForecasterUEAClassification(UEAClassificationExp, SparseTransformerForecasterParameters):
    model_type: str = "SparseTransformerForecaster"

    def _init_model(self):
        self.model = SparseTransformerForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            patch_size=self.patch_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            local_window=self.local_window,
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
class SparseTransformerForecasterAnomalyDetection(AnomalyDetectionExp, SparseTransformerForecasterParameters):
    model_type: str = "SparseTransformerForecaster"

    def _init_model(self):
        self.model = SparseTransformerForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            patch_size=self.patch_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            local_window=self.local_window,
            stride=self.stride,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class SparseTransformerForecasterImputation(ImputationExp, SparseTransformerForecasterParameters):
    model_type: str = "SparseTransformerForecaster"

    def _init_model(self):
        self.model = SparseTransformerForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            patch_size=self.patch_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            local_window=self.local_window,
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
