from dataclasses import dataclass
import sys

import torch
from ..model import FITS
from . import (
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
    AnomalyDetectionExp,
)


@dataclass
class FITSParameters:
    individual: bool = True
    cut_freq : int = 8

@dataclass
class FITSForecast(ForecastExp, FITSParameters):
    model_type: str = "FITS"

    def _init_model(self):
        self.model = FITS(
            seq_len=self.windows,
            pred_len=self.pred_len,
            individual=self.individual,
            enc_in=self.dataset.num_features,
            cut_freq=self.cut_freq,
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc):
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
        pred, low = self.model(
            batch_x
        )  # torch.Size([batch_size, output_length, num_nodes])
        return pred[:, -self.pred_len:, :], batch_y

class FITSClassModel(FITS):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.projection = torch.nn.Linear(
           int( kwargs.get("enc_in") * self.seq_len), self.num_classes)

    def forward(self, x):
        enc_out, _ = super().forward(x) # B, O, N
        enc_out = enc_out[:, -self.pred_len:, :].reshape(x.shape[0], -1)
        output = self.projection(torch.relu(enc_out))
        return output




@dataclass
class FITSUEAClassification(UEAClassificationExp, FITSParameters):
    model_type: str = "FITS"

    def _init_model(self):
        self.model = FITSClassModel(
            seq_len=self.windows,
            pred_len=self.windows,
            individual=self.individual,
            enc_in=self.dataset.num_features,
            cut_freq=self.cut_freq,
            num_classes=self.dataset.num_classes,
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(
            batch_x
        )  # torch.Size([batch_size, output_length, num_nodes])
        return outputs, batch_y.long().squeeze(-1)
