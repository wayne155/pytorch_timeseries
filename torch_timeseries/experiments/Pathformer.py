from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch

from ..model.Pathformer import Pathformer
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class PathformerParameters:
    patch_sizes: List[int] = field(default_factory=lambda: [4, 8, 16])
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    revin: bool = True


@dataclass
class PathformerForecast(ForecastExp, PathformerParameters):
    model_type: str = "Pathformer"

    def _init_model(self):
        self.model = Pathformer(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            patch_sizes=self.patch_sizes,
            d_model=self.d_model,
            n_heads=self.n_heads,
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
class PathformerUEAClassification(UEAClassificationExp, PathformerParameters):
    model_type: str = "Pathformer"

    def _init_model(self):
        self.model = Pathformer(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            patch_sizes=self.patch_sizes,
            d_model=self.d_model,
            n_heads=self.n_heads,
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
class PathformerAnomalyDetection(AnomalyDetectionExp, PathformerParameters):
    model_type: str = "Pathformer"

    def _init_model(self):
        self.model = Pathformer(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            patch_sizes=self.patch_sizes,
            d_model=self.d_model,
            n_heads=self.n_heads,
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
class PathformerImputation(ImputationExp, PathformerParameters):
    model_type: str = "Pathformer"

    def _init_model(self):
        self.model = Pathformer(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            patch_sizes=self.patch_sizes,
            d_model=self.d_model,
            n_heads=self.n_heads,
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
