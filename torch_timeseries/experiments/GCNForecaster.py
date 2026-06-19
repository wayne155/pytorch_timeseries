from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.GCNForecaster import GCNForecaster
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class GCNForecasterParameters:
    d_model: int = 64
    e_layers: int = 3
    d_emb: int = 10
    k_hops: int = 2
    kernel_size: int = 3
    dropout: float = 0.1
    revin: bool = True


@dataclass
class GCNForecasterForecast(ForecastExp, GCNForecasterParameters):
    model_type: str = "GCNForecaster"

    def _init_model(self):
        self.model = GCNForecaster(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            e_layers=self.e_layers,
            d_emb=self.d_emb,
            k_hops=self.k_hops,
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
class GCNForecasterUEAClassification(UEAClassificationExp, GCNForecasterParameters):
    model_type: str = "GCNForecaster"

    def _init_model(self):
        self.model = GCNForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            e_layers=self.e_layers,
            d_emb=self.d_emb,
            k_hops=self.k_hops,
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
class GCNForecasterAnomalyDetection(AnomalyDetectionExp, GCNForecasterParameters):
    model_type: str = "GCNForecaster"

    def _init_model(self):
        self.model = GCNForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            e_layers=self.e_layers,
            d_emb=self.d_emb,
            k_hops=self.k_hops,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class GCNForecasterImputation(ImputationExp, GCNForecasterParameters):
    model_type: str = "GCNForecaster"

    def _init_model(self):
        self.model = GCNForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            e_layers=self.e_layers,
            d_emb=self.d_emb,
            k_hops=self.k_hops,
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
