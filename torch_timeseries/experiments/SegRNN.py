from dataclasses import dataclass

import torch

from ..model import SegRNN
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class SegRNNParameters:
    d_model: int = 512
    seg_len: int = 48
    dropout: float = 0.5


@dataclass
class SegRNNForecast(ForecastExp, SegRNNParameters):
    model_type: str = "SegRNN"

    def _init_model(self):
        self.model = SegRNN(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            seg_len=self.seg_len,
            dropout=self.dropout,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_y


@dataclass
class SegRNNUEAClassification(UEAClassificationExp, SegRNNParameters):
    model_type: str = "SegRNN"

    def _init_model(self):
        self.model = SegRNN(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            seg_len=self.seg_len,
            dropout=self.dropout,
            output_prob=self.dataset.num_classes,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_y.long().squeeze(-1)


@dataclass
class SegRNNAnomalyDetection(AnomalyDetectionExp, SegRNNParameters):
    model_type: str = "SegRNN"

    def _init_model(self):
        self.model = SegRNN(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            seg_len=self.seg_len,
            dropout=self.dropout,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_x


@dataclass
class SegRNNImputation(ImputationExp, SegRNNParameters):
    model_type: str = "SegRNN"

    def _init_model(self):
        self.model = SegRNN(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            seg_len=self.seg_len,
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
