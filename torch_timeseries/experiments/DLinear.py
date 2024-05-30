from dataclasses import dataclass
import sys

import torch
from ..model import DLinear
from . import (
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
    AnomalyDetectionExp,
)


@dataclass
class DLinearParameters:
    individual: bool = False


@dataclass
class DLinearForecast(ForecastExp, DLinearParameters):
    model_type: str = "DLinear"

    def _init_model(self):
        self.model = DLinear(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            individual=self.individual,
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
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
        outputs = self.model(
            batch_x
        )  # torch.Size([batch_size, output_length, num_nodes])
        return outputs, batch_y


@dataclass
class DLinearUEAClassification(UEAClassificationExp, DLinearParameters):
    model_type: str = "DLinear"

    def _init_model(self):
        self.model = DLinear(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            individual=self.individual,
            output_prob=self.dataset.num_classes
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


@dataclass
class DLinearAnomalyDetection(AnomalyDetectionExp, DLinearParameters):
    model_type: str = "DLinear"

    def _init_model(self):
        self.model = DLinear(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            individual=self.individual,
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        # inputs:
            # batch_x: (B, T, N)
            # origin_x: (B, T, N)
        # ouputs:
        # - pred: (B, O, N)
        # - label: (B, O, N)
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(
            batch_x
        )  # torch.Size([batch_size, output_length, num_nodes])
        return outputs, batch_x


@dataclass
class DLinearImputation(ImputationExp, DLinearParameters):
    model_type: str = "DLinear"

    def _init_model(self):
        self.model = DLinear(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            individual=self.individual,
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
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_masked_x = batch_masked_x.to(self.device, dtype=torch.float32)
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(
            batch_masked_x
        )  # torch.Size([batch_size, output_length, num_nodes])
        return outputs, batch_x


