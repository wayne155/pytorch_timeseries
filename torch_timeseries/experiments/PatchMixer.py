from dataclasses import dataclass

import torch

from ..model.PatchMixer import PatchMixer
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class PatchMixerParameters:
    patch_len: int = 16
    stride: int = 8
    d_model: int = 64
    depth: int = 3
    dropout: float = 0.1
    revin: bool = True


@dataclass
class PatchMixerForecast(ForecastExp, PatchMixerParameters):
    model_type: str = "PatchMixer"

    def _init_model(self):
        self.model = PatchMixer(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            depth=self.depth,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_y


@dataclass
class PatchMixerUEAClassification(UEAClassificationExp, PatchMixerParameters):
    model_type: str = "PatchMixer"

    def _init_model(self):
        self.model = PatchMixer(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            depth=self.depth,
            dropout=self.dropout,
            revin=False,
            output_prob=self.dataset.num_classes,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_y.long().squeeze(-1)


@dataclass
class PatchMixerAnomalyDetection(AnomalyDetectionExp, PatchMixerParameters):
    model_type: str = "PatchMixer"

    def _init_model(self):
        self.model = PatchMixer(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            depth=self.depth,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_x


@dataclass
class PatchMixerImputation(ImputationExp, PatchMixerParameters):
    model_type: str = "PatchMixer"

    def _init_model(self):
        self.model = PatchMixer(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            depth=self.depth,
            dropout=self.dropout,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(
        self, batch_masked_x, batch_x, batch_origin_x, batch_mask, batch_x_date_enc
    ):
        batch_masked_x = batch_masked_x.to(self.device, dtype=torch.float32)
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        return self.model(batch_masked_x), batch_x
