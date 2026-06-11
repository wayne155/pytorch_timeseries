from dataclasses import dataclass, field
import sys

import torch

from ..model import TimesNet
from . import (
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
    AnomalyDetectionExp,
)


@dataclass
class TimesNetParameters:
    n_heads: int = 8
    e_layers: int = 2
    label_len: int = 48
    d_model: int = 512
    e_layers: int = 2
    d_ff: int = 512  # out of memoery with d_ff = 2048
    num_kernels: int = 6
    top_k: int = 5
    dropout: float = 0.0
    embed: str = "timeF"

@dataclass
class TimesNetForecast(ForecastExp, TimesNetParameters):
    model_type: str = "TimesNet"
    
    def _init_model(self):
        self.model = TimesNet(
            seq_len=self.windows, 
            label_len=self.label_len,
            pred_len=self.pred_len, 
            e_layers=self.e_layers, 
            d_ff=self.d_ff,
            num_kernels=self.num_kernels,
            top_k=self.top_k,
            d_model=self.d_model,
            embed=self.embed,
            enc_in=self.dataset.num_features,
            freq=self.dataset.freq,
            dropout=self.dropout,
            c_out=self.dataset.num_features,
            task_name="long_term_forecast",
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
        
        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset.num_features]
        ).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len :, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        dec_inp_date_enc = torch.cat(
            [batch_x_date_enc[:, -self.label_len :, :], batch_y_date_enc], dim=1
        )
        outputs = self.model(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc)  # torch.Size([batch_size, output_length, num_nodes])
            
        return outputs, batch_y




@dataclass
class TimesNetAnomalyDetection(AnomalyDetectionExp, TimesNetParameters):
    model_type: str = "TimesNet"

    def _init_model(self):
        self.model = TimesNet(
            seq_len=self.windows,
            label_len=self.label_len,
            pred_len=0,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            num_kernels=self.num_kernels,
            top_k=self.top_k,
            d_model=self.d_model,
            embed=self.embed,
            enc_in=self.dataset.num_features,
            freq="h",
            dropout=self.dropout,
            c_out=self.dataset.num_features,
            task_name="anomaly_detection",
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x, None, None, None)
        return outputs, batch_x


@dataclass
class TimesNetImputation(ImputationExp, TimesNetParameters):
    model_type: str = "TimesNet"

    def _init_model(self):
        self.model = TimesNet(
            seq_len=self.windows,
            label_len=self.label_len,
            pred_len=0,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            num_kernels=self.num_kernels,
            top_k=self.top_k,
            d_model=self.d_model,
            embed=self.embed,
            enc_in=self.dataset.num_features,
            freq=self.dataset.freq,
            dropout=self.dropout,
            c_out=self.dataset.num_features,
            task_name="imputation",
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
        batch_x_date_enc = batch_x_date_enc.to(self.device, dtype=torch.float32)
        batch_mask = batch_mask.to(self.device)
        outputs = self.model(batch_masked_x, batch_x_date_enc, None, None, batch_mask)
        return outputs, batch_x


@dataclass
class TimesNetUEAClassification(UEAClassificationExp, TimesNetParameters):
    model_type: str = "TimesNet"

    def _init_model(self):
        self.model = TimesNet(
            seq_len=self.windows, 
            label_len=self.label_len,
            pred_len=0, 
            e_layers=self.e_layers, 
            d_ff=self.d_ff,
            num_kernels=self.num_kernels,
            top_k=self.top_k,
            d_model=self.d_model,
            embed='timeF',
            enc_in=self.dataset.num_features,
            freq='h',
            dropout=self.dropout,
            c_out=self.dataset.num_features,
            num_class=self.dataset.num_classes,
            task_name="classification",
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
        padding_masks = padding_masks.to(self.device, dtype=torch.float32)
        
        outputs = self.model(batch_x, padding_masks, None, None)  # torch.Size([batch_size, output_length, num_nodes])
        return outputs, batch_y.long().squeeze(-1)
