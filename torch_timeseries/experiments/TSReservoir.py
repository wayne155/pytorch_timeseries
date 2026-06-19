from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.TSReservoir import TSReservoir
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class TSReservoirParameters:
    d_res: int = 256
    spectral_radius: float = 0.9
    input_scale: float = 0.1
    pool_states: bool = True
    revin: bool = True


@dataclass
class TSReservoirForecast(ForecastExp, TSReservoirParameters):
    model_type: str = "TSReservoir"

    def _init_model(self):
        self.model = TSReservoir(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            d_res=self.d_res,
            spectral_radius=self.spectral_radius,
            input_scale=self.input_scale,
            pool_states=self.pool_states,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class TSReservoirUEAClassification(UEAClassificationExp, TSReservoirParameters):
    model_type: str = "TSReservoir"

    def _init_model(self):
        self.model = TSReservoir(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_res=self.d_res,
            spectral_radius=self.spectral_radius,
            input_scale=self.input_scale,
            pool_states=self.pool_states,
            revin=False,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class TSReservoirAnomalyDetection(AnomalyDetectionExp, TSReservoirParameters):
    model_type: str = "TSReservoir"

    def _init_model(self):
        self.model = TSReservoir(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_res=self.d_res,
            spectral_radius=self.spectral_radius,
            input_scale=self.input_scale,
            pool_states=self.pool_states,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class TSReservoirImputation(ImputationExp, TSReservoirParameters):
    model_type: str = "TSReservoir"

    def _init_model(self):
        self.model = TSReservoir(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_res=self.d_res,
            spectral_radius=self.spectral_radius,
            input_scale=self.input_scale,
            pool_states=self.pool_states,
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
