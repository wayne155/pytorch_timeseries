from dataclasses import dataclass

import torch

from ..model.VanillaTransformer import VanillaTransformer
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class VanillaTransformerParameters:
    d_model: int = 256
    n_heads: int = 4
    e_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    revin: bool = True


@dataclass
class VanillaTransformerForecast(ForecastExp, VanillaTransformerParameters):
    model_type: str = "VanillaTransformer"

    def _init_model(self):
        self.model = VanillaTransformer(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_y


@dataclass
class VanillaTransformerUEAClassification(UEAClassificationExp, VanillaTransformerParameters):
    model_type: str = "VanillaTransformer"

    def _init_model(self):
        self.model = VanillaTransformer(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            revin=False,
            output_prob=self.dataset.num_classes,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_y.long().squeeze(-1)


@dataclass
class VanillaTransformerAnomalyDetection(AnomalyDetectionExp, VanillaTransformerParameters):
    model_type: str = "VanillaTransformer"

    def _init_model(self):
        self.model = VanillaTransformer(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_x


@dataclass
class VanillaTransformerImputation(ImputationExp, VanillaTransformerParameters):
    model_type: str = "VanillaTransformer"

    def _init_model(self):
        self.model = VanillaTransformer(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            revin=self.revin,
        ).to(self.device)

    def _process_one_batch(
        self, batch_masked_x, batch_x, batch_origin_x, batch_mask, batch_x_date_enc
    ):
        batch_masked_x = batch_masked_x.to(self.device, dtype=torch.float32)
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        return self.model(batch_masked_x), batch_x
