from dataclasses import dataclass

import torch

from ..model import NHiTS
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class NHiTSParameters:
    n_stacks: int = 3
    n_blocks: int = 1
    n_theta: int = 512
    mlp_units: int = 512
    n_layers: int = 2
    dropout: float = 0.1


@dataclass
class NHiTSForecast(ForecastExp, NHiTSParameters):
    model_type: str = "NHiTS"

    def _init_model(self):
        self.model = NHiTS(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            n_stacks=self.n_stacks,
            n_blocks=self.n_blocks,
            n_theta=self.n_theta,
            mlp_units=self.mlp_units,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_y


@dataclass
class NHiTSUEAClassification(UEAClassificationExp, NHiTSParameters):
    model_type: str = "NHiTS"

    def _init_model(self):
        self.model = NHiTS(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            n_stacks=self.n_stacks,
            n_blocks=self.n_blocks,
            n_theta=self.n_theta,
            mlp_units=self.mlp_units,
            n_layers=self.n_layers,
            dropout=self.dropout,
            output_prob=self.dataset.num_classes,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_y.long().squeeze(-1)


@dataclass
class NHiTSAnomalyDetection(AnomalyDetectionExp, NHiTSParameters):
    model_type: str = "NHiTS"

    def _init_model(self):
        self.model = NHiTS(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            n_stacks=self.n_stacks,
            n_blocks=self.n_blocks,
            n_theta=self.n_theta,
            mlp_units=self.mlp_units,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_x


@dataclass
class NHiTSImputation(ImputationExp, NHiTSParameters):
    model_type: str = "NHiTS"

    def _init_model(self):
        self.model = NHiTS(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            n_stacks=self.n_stacks,
            n_blocks=self.n_blocks,
            n_theta=self.n_theta,
            mlp_units=self.mlp_units,
            n_layers=self.n_layers,
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
