
from dataclasses import dataclass, field
import sys

import torch
from ..model import CATS
from . import (
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
    AnomalyDetectionExp,
)

@dataclass
class CATSParameters:
    d_model: int = 256
    e_layers: int = 3
    d_layers:int = 3
    d_ff: int = 512  # out of memoery with d_ff = 2048
    dropout: float = 0.0
    n_heads : int = 8
    patch_len : int = 16
    stride : int = 8
    label_len :int = 48
    QAM_start :float = 0.1
    QAM_end :float= 0.3
    
    query_independence: bool = True
    store_attn: bool = True
    padding_patch : str = 'end'


@dataclass
class CATSForecast(ForecastExp, CATSParameters):
    model_type: str = "CATS"

    def _init_model(self):
        self.model = CATS(
            seq_len=self.windows,
            pred_len=self.pred_len,
            dec_in=self.dataset.num_features,
            d_layers=self.d_layers,
            n_heads=self.n_heads,
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout=self.dropout,
            query_independence=self.query_independence,
            patch_len=self.patch_len,
            stride=self.stride,
            padding_patch=self.padding_patch,
            store_attn=self.store_attn,
            QAM_start=self.QAM_start,
            QAM_end=self.QAM_end
            )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_x = batch_x.to(self.device).float()
        batch_y = batch_y.to(self.device).float()
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()

        # no decoder input
        # label_len = 1
        outputs = self.model(batch_x)
        return outputs, batch_y



