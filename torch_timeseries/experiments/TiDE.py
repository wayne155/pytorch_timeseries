from dataclasses import dataclass

import torch

from ..model import TiDE
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class TiDEParameters:
    hidden_size: int = 256
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    decoder_output_dim: int = 8
    dropout: float = 0.3


@dataclass
class TiDEForecast(ForecastExp, TiDEParameters):
    model_type: str = "TiDE"

    def _init_model(self):
        self.model = TiDE(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            hidden_size=self.hidden_size,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            decoder_output_dim=self.decoder_output_dim,
            dropout=self.dropout,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_y


@dataclass
class TiDEUEAClassification(UEAClassificationExp, TiDEParameters):
    model_type: str = "TiDE"

    def _init_model(self):
        self.model = TiDE(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            hidden_size=self.hidden_size,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            decoder_output_dim=self.decoder_output_dim,
            dropout=self.dropout,
            output_prob=self.dataset.num_classes,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_y.long().squeeze(-1)


@dataclass
class TiDEAnomalyDetection(AnomalyDetectionExp, TiDEParameters):
    model_type: str = "TiDE"

    def _init_model(self):
        self.model = TiDE(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            hidden_size=self.hidden_size,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            decoder_output_dim=self.decoder_output_dim,
            dropout=self.dropout,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_x


@dataclass
class TiDEImputation(ImputationExp, TiDEParameters):
    model_type: str = "TiDE"

    def _init_model(self):
        self.model = TiDE(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            hidden_size=self.hidden_size,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            decoder_output_dim=self.decoder_output_dim,
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
        return self.model(batch_masked_x), batch_x
