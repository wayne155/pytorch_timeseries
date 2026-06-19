from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.TFT import TFT
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class TFTParameters:
    d_model: int = 128
    n_heads: int = 4
    num_lstm_layers: int = 2
    dropout: float = 0.1


@dataclass
class TFTForecast(ForecastExp, TFTParameters):
    model_type: str = "TFT"

    def _init_model(self):
        self.model = TFT(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_lstm_layers=self.num_lstm_layers,
            dropout=self.dropout,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class TFTUEAClassification(UEAClassificationExp, TFTParameters):
    model_type: str = "TFT"

    def _init_model(self):
        self.model = TFT(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_lstm_layers=self.num_lstm_layers,
            dropout=self.dropout,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class TFTAnomalyDetection(AnomalyDetectionExp, TFTParameters):
    model_type: str = "TFT"

    def _init_model(self):
        self.model = TFT(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_lstm_layers=self.num_lstm_layers,
            dropout=self.dropout,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class TFTImputation(ImputationExp, TFTParameters):
    model_type: str = "TFT"

    def _init_model(self):
        self.model = TFT(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_lstm_layers=self.num_lstm_layers,
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
