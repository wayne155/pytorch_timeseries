from dataclasses import dataclass

import torch

from ..model import NLinear
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class NLinearParameters:
    individual: bool = False


@dataclass
class NLinearForecast(ForecastExp, NLinearParameters):
    model_type: str = "NLinear"

    def _init_model(self):
        self.model = NLinear(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            individual=self.individual,
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch):
        batch_x, batch_y = batch.x, batch.y
        origin_x, origin_y = batch.x_raw, batch.y_raw
        batch_x_date_enc, batch_y_date_enc = batch.x_time_feature, batch.y_time_feature
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class NLinearUEAClassification(UEAClassificationExp, NLinearParameters):
    model_type: str = "NLinear"

    def _init_model(self):
        self.model = NLinear(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            individual=self.individual,
            output_prob=self.dataset.num_classes,
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class NLinearAnomalyDetection(AnomalyDetectionExp, NLinearParameters):
    model_type: str = "NLinear"

    def _init_model(self):
        self.model = NLinear(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            individual=self.individual,
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class NLinearImputation(ImputationExp, NLinearParameters):
    model_type: str = "NLinear"

    def _init_model(self):
        self.model = NLinear(
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
        batch_masked_x = batch_masked_x.to(self.device, dtype=torch.float32)
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_masked_x)
        return outputs, batch_x
