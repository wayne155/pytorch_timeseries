from dataclasses import dataclass
import sys

import torch
from ..model import FreTS
from . import (
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
    AnomalyDetectionExp,
)


@dataclass
class FreTSParameters:
    channel_independence: bool = True

@dataclass
class FreTSForecast(ForecastExp, FreTSParameters):
    model_type: str = "FreTS"

    def _init_model(self):
        self.model = FreTS(
            seq_len=self.windows,
            pred_len=self.pred_len,
            channel_independence=self.channel_independence,
            enc_in=self.dataset.num_features,
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch):
        batch_x, batch_y = batch.x, batch.y
        origin_x, origin_y = batch.x_raw, batch.y_raw
        batch_x_date_enc, batch_y_date_enc = batch.x_time_feature, batch.y_time_feature
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()
        pred = self.model(
            batch_x
        )  # torch.Size([batch_size, output_length, num_nodes])
        return pred[:, -self.pred_len:, :], batch_y

class FreTSClassModel(FreTS):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.projection = torch.nn.Linear(
           int( kwargs.get("enc_in") * self.seq_length), self.num_classes)

    def forward(self, x):
        enc_out = super().forward(x) # B, O, N
        enc_out = enc_out[:, -self.pre_length:, :].reshape(x.shape[0], -1)
        output = self.projection(torch.relu(enc_out))
        return output




@dataclass
class FreTSUEAClassification(UEAClassificationExp, FreTSParameters):
    model_type: str = "FreTS"

    def _init_model(self):
        self.model = FreTSClassModel(
            seq_len=self.windows,
            pred_len=self.windows,
            channel_independence=self.channel_independence,
            enc_in=self.dataset.num_features,
            num_classes=self.dataset.num_classes,
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class FreTSAnomalyDetection(AnomalyDetectionExp, FreTSParameters):
    model_type: str = "FreTS"

    def _init_model(self):
        self.model = FreTS(
            seq_len=self.windows,
            pred_len=self.windows,
            channel_independence=self.channel_independence,
            enc_in=self.dataset.num_features,
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class FreTSImputation(ImputationExp, FreTSParameters):
    model_type: str = "FreTS"

    def _init_model(self):
        self.model = FreTS(
            seq_len=self.windows,
            pred_len=self.windows,
            channel_independence=self.channel_independence,
            enc_in=self.dataset.num_features,
        )
        self.model = self.model.to(self.device)

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
