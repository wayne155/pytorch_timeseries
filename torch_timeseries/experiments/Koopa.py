from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..model.Koopa import Koopa
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class KoopaParameters:
    seg_len: int = 10
    d_model: int = 128
    n_ff: Optional[int] = None
    top_k: int = 5
    revin: bool = True
    dropout: float = 0.0


@dataclass
class KoopaForecast(ForecastExp, KoopaParameters):
    model_type: str = "Koopa"

    def _init_model(self):
        self.model = Koopa(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            seg_len=self.seg_len,
            d_model=self.d_model,
            n_ff=self.n_ff,
            top_k=self.top_k,
            revin=self.revin,
            dropout=self.dropout,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class KoopaUEAClassification(UEAClassificationExp, KoopaParameters):
    model_type: str = "Koopa"

    def _init_model(self):
        self.model = Koopa(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            seg_len=self.seg_len,
            d_model=self.d_model,
            n_ff=self.n_ff,
            top_k=self.top_k,
            revin=False,
            dropout=self.dropout,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class KoopaAnomalyDetection(AnomalyDetectionExp, KoopaParameters):
    model_type: str = "Koopa"

    def _init_model(self):
        self.model = Koopa(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            seg_len=self.seg_len,
            d_model=self.d_model,
            n_ff=self.n_ff,
            top_k=self.top_k,
            revin=self.revin,
            dropout=self.dropout,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class KoopaImputation(ImputationExp, KoopaParameters):
    model_type: str = "Koopa"

    def _init_model(self):
        self.model = Koopa(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            seg_len=self.seg_len,
            d_model=self.d_model,
            n_ff=self.n_ff,
            top_k=self.top_k,
            revin=self.revin,
            dropout=self.dropout,
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
