from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.MoEForecaster import MoEForecaster
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class MoEForecasterParameters:
    n_experts: int = 8
    k_active: int = 2
    d_router: int = 32
    expert_type: str = "linear"
    d_ff: int = 128
    dropout: float = 0.1
    revin: bool = True


@dataclass
class MoEForecasterForecast(ForecastExp, MoEForecasterParameters):
    model_type: str = "MoEForecaster"

    def _init_model(self):
        self.model = MoEForecaster(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            n_experts=self.n_experts,
            k_active=self.k_active,
            d_router=self.d_router,
            expert_type=self.expert_type,
            d_ff=self.d_ff,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class MoEForecasterUEAClassification(UEAClassificationExp, MoEForecasterParameters):
    model_type: str = "MoEForecaster"

    def _init_model(self):
        self.model = MoEForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            n_experts=self.n_experts,
            k_active=self.k_active,
            d_router=self.d_router,
            expert_type=self.expert_type,
            d_ff=self.d_ff,
            dropout=self.dropout,
            revin=False,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class MoEForecasterAnomalyDetection(AnomalyDetectionExp, MoEForecasterParameters):
    model_type: str = "MoEForecaster"

    def _init_model(self):
        self.model = MoEForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            n_experts=self.n_experts,
            k_active=self.k_active,
            d_router=self.d_router,
            expert_type=self.expert_type,
            d_ff=self.d_ff,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class MoEForecasterImputation(ImputationExp, MoEForecasterParameters):
    model_type: str = "MoEForecaster"

    def _init_model(self):
        self.model = MoEForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            n_experts=self.n_experts,
            k_active=self.k_active,
            d_router=self.d_router,
            expert_type=self.expert_type,
            d_ff=self.d_ff,
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
