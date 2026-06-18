from dataclasses import dataclass

import torch

from ..model import TimeMixer
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class TimeMixerParameters:
    n_heads: int = 4
    d_model: int = 32
    e_layers: int = 3
    dropout: float = 0.1
    down_sampling_window: int = 2
    down_sampling_layers: int = 3


@dataclass
class TimeMixerForecast(ForecastExp, TimeMixerParameters):
    model_type: str = "TimeMixer"

    def _init_model(self):
        self.model = TimeMixer(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            n_heads=self.n_heads,
            d_model=self.d_model,
            e_layers=self.e_layers,
            dropout=self.dropout,
            down_sampling_window=self.down_sampling_window,
            down_sampling_layers=self.down_sampling_layers,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_y


@dataclass
class TimeMixerUEAClassification(UEAClassificationExp, TimeMixerParameters):
    model_type: str = "TimeMixer"

    def _init_model(self):
        self.model = TimeMixer(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            n_heads=self.n_heads,
            d_model=self.d_model,
            e_layers=self.e_layers,
            dropout=self.dropout,
            down_sampling_window=self.down_sampling_window,
            down_sampling_layers=self.down_sampling_layers,
            output_prob=self.dataset.num_classes,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_y.long().squeeze(-1)


@dataclass
class TimeMixerAnomalyDetection(AnomalyDetectionExp, TimeMixerParameters):
    model_type: str = "TimeMixer"

    def _init_model(self):
        self.model = TimeMixer(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            n_heads=self.n_heads,
            d_model=self.d_model,
            e_layers=self.e_layers,
            dropout=self.dropout,
            down_sampling_window=self.down_sampling_window,
            down_sampling_layers=self.down_sampling_layers,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_x


@dataclass
class TimeMixerImputation(ImputationExp, TimeMixerParameters):
    model_type: str = "TimeMixer"

    def _init_model(self):
        self.model = TimeMixer(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            n_heads=self.n_heads,
            d_model=self.d_model,
            e_layers=self.e_layers,
            dropout=self.dropout,
            down_sampling_window=self.down_sampling_window,
            down_sampling_layers=self.down_sampling_layers,
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
        return self.model(batch_masked_x), batch_x
