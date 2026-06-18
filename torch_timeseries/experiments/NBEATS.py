from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from ..model.NBEATS import NBEATS
from . import (
    AnomalyDetectionExp,
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
)


@dataclass
class NBEATSParameters:
    stack_types: List[str] = field(default_factory=lambda: ["generic", "generic", "generic"])
    num_blocks: int = 3
    hidden_size: int = 256
    expansion_coefficient_dim: int = 32
    degree_of_polynomial: int = 3
    num_harmonics: int = 1


@dataclass
class NBEATSForecast(ForecastExp, NBEATSParameters):
    model_type: str = "NBEATS"

    def _init_model(self):
        self.model = NBEATS(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            stack_types=self.stack_types,
            num_blocks=self.num_blocks,
            hidden_size=self.hidden_size,
            expansion_coefficient_dim=self.expansion_coefficient_dim,
            degree_of_polynomial=self.degree_of_polynomial,
            num_harmonics=self.num_harmonics,
        ).to(self.device)

    def _process_one_batch(self, batch):
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y


@dataclass
class NBEATSUEAClassification(UEAClassificationExp, NBEATSParameters):
    model_type: str = "NBEATS"

    def _init_model(self):
        self.model = NBEATS(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            stack_types=self.stack_types,
            num_blocks=self.num_blocks,
            hidden_size=self.hidden_size,
            expansion_coefficient_dim=self.expansion_coefficient_dim,
            degree_of_polynomial=self.degree_of_polynomial,
            num_harmonics=self.num_harmonics,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class NBEATSAnomalyDetection(AnomalyDetectionExp, NBEATSParameters):
    model_type: str = "NBEATS"

    def _init_model(self):
        self.model = NBEATS(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            stack_types=self.stack_types,
            num_blocks=self.num_blocks,
            hidden_size=self.hidden_size,
            expansion_coefficient_dim=self.expansion_coefficient_dim,
            degree_of_polynomial=self.degree_of_polynomial,
            num_harmonics=self.num_harmonics,
        ).to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x)
        return outputs, batch_x


@dataclass
class NBEATSImputation(ImputationExp, NBEATSParameters):
    model_type: str = "NBEATS"

    def _init_model(self):
        self.model = NBEATS(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            stack_types=self.stack_types,
            num_blocks=self.num_blocks,
            hidden_size=self.hidden_size,
            expansion_coefficient_dim=self.expansion_coefficient_dim,
            degree_of_polynomial=self.degree_of_polynomial,
            num_harmonics=self.num_harmonics,
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
