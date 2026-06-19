from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.ModernTCN import ModernTCN
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class ModernTCNParameters:
    patch_size: int = 8
    patch_stride: int = 4
    d_model: int = 128
    kernel_size: int = 51
    e_layers: int = 3
    d_ff_ratio: int = 4
    dropout: float = 0.05
    revin: bool = True


@dataclass
class ModernTCNForecast(ForecastExp, ModernTCNParameters):
    model_type: str = "ModernTCN"

    def _init_model(self):
        self.model = ModernTCN(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            d_model=self.d_model,
            kernel_size=self.kernel_size,
            e_layers=self.e_layers,
            d_ff_ratio=self.d_ff_ratio,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class ModernTCNUEAClassification(UEAClassificationExp, ModernTCNParameters):
    model_type: str = "ModernTCN"

    def _init_model(self):
        self.model = ModernTCN(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            d_model=self.d_model,
            kernel_size=self.kernel_size,
            e_layers=self.e_layers,
            d_ff_ratio=self.d_ff_ratio,
            dropout=self.dropout,
            revin=False,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class ModernTCNAnomalyDetection(AnomalyDetectionExp, ModernTCNParameters):
    model_type: str = "ModernTCN"

    def _init_model(self):
        self.model = ModernTCN(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            d_model=self.d_model,
            kernel_size=self.kernel_size,
            e_layers=self.e_layers,
            d_ff_ratio=self.d_ff_ratio,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class ModernTCNImputation(ImputationExp, ModernTCNParameters):
    model_type: str = "ModernTCN"

    def _init_model(self):
        self.model = ModernTCN(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            d_model=self.d_model,
            kernel_size=self.kernel_size,
            e_layers=self.e_layers,
            d_ff_ratio=self.d_ff_ratio,
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
