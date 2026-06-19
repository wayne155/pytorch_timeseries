from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.iMamba import iMamba
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class iMambaParameters:
    d_model: int = 128
    d_state: int = 16
    e_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.05
    revin: bool = True


@dataclass
class iMambaForecast(ForecastExp, iMambaParameters):
    model_type: str = "iMamba"

    def _init_model(self):
        self.model = iMamba(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            d_state=self.d_state,
            e_layers=self.e_layers,
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
class iMambaUEAClassification(UEAClassificationExp, iMambaParameters):
    model_type: str = "iMamba"

    def _init_model(self):
        self.model = iMamba(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            d_state=self.d_state,
            e_layers=self.e_layers,
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
class iMambaAnomalyDetection(AnomalyDetectionExp, iMambaParameters):
    model_type: str = "iMamba"

    def _init_model(self):
        self.model = iMamba(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            d_state=self.d_state,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class iMambaImputation(ImputationExp, iMambaParameters):
    model_type: str = "iMamba"

    def _init_model(self):
        self.model = iMamba(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            d_state=self.d_state,
            e_layers=self.e_layers,
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
