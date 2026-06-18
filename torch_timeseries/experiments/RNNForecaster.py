from dataclasses import dataclass

import torch

from ..model.RNNForecaster import RNNForecaster
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class RNNForecasterParameters:
    hidden_size: int = 64
    num_layers: int = 2
    rnn_type: str = "gru"
    dropout: float = 0.1
    bidirectional: bool = False
    revin: bool = True


@dataclass
class RNNForecast(ForecastExp, RNNForecasterParameters):
    model_type: str = "RNN"

    def _init_model(self):
        self.model = RNNForecaster(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            rnn_type=self.rnn_type,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_y


@dataclass
class RNNUEAClassification(UEAClassificationExp, RNNForecasterParameters):
    model_type: str = "RNN"

    def _init_model(self):
        self.model = RNNForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            rnn_type=self.rnn_type,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            revin=False,
            output_prob=self.dataset.num_classes,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_y.long().squeeze(-1)


@dataclass
class RNNAnomalyDetection(AnomalyDetectionExp, RNNForecasterParameters):
    model_type: str = "RNN"

    def _init_model(self):
        self.model = RNNForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            rnn_type=self.rnn_type,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_x


@dataclass
class RNNImputation(ImputationExp, RNNForecasterParameters):
    model_type: str = "RNN"

    def _init_model(self):
        self.model = RNNForecaster(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            rnn_type=self.rnn_type,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(
        self, batch_masked_x, batch_x, batch_origin_x, batch_mask, batch_x_date_enc
    ):
        batch_masked_x = batch_masked_x.to(self.device, dtype=torch.float32)
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        return self.model(batch_masked_x), batch_x
