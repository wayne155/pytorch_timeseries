from dataclasses import dataclass
import sys

import torch
from ..model import PatchTST
from . import (
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
    AnomalyDetectionExp,
)


@dataclass
class PatchTSTParameters:
    d_model: int = 512
    e_layers: int = 2
    d_ff: int = 512  # out of memoery with d_ff = 2048
    dropout: float = 0.0
    n_heads : int = 8
    patch_len : int = 16
    stride : int = 8
    label_len :int = 48


@dataclass
class PatchTSTForecast(ForecastExp, PatchTSTParameters):
    model_type: str = "PatchTST"

    def _init_model(self):
        self.model = PatchTST(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            n_heads=self.n_heads,
            dropout=self.dropout,
            e_layers=self.e_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            patch_len=self.patch_len,
            stride=self.stride,
            task_name="long_term_forecast",
            num_class=0
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
        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset.num_features]
        ).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len:, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        dec_inp_date_enc = torch.cat(
            [batch_x_date_enc[:, -self.label_len:, :], batch_y_date_enc], dim=1
        )
        outputs = self.model(batch_x, batch_x_date_enc,
                             dec_inp, dec_inp_date_enc)
        return outputs, batch_y


@dataclass
class PatchTSTUEAClassification(UEAClassificationExp, PatchTSTParameters):
    model_type: str = "PatchTST"

    def _init_model(self):
        self.model = PatchTST(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            n_heads=self.n_heads,
            dropout=self.dropout,
            e_layers=self.e_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            patch_len=self.patch_len,
            stride=self.stride,
            task_name="classification",
            num_class=self.dataset.num_classes
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
            batch_x, padding_masks, None, None
        )  # torch.Size([batch_size, output_length, num_nodes])
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class PatchTSTAnomalyDetection(AnomalyDetectionExp, PatchTSTParameters):
    model_type: str = "PatchTST"

    def _init_model(self):
        self.model = PatchTST(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            n_heads=self.n_heads,
            dropout=self.dropout,
            e_layers=self.e_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            patch_len=self.patch_len,
            stride=self.stride,
            task_name="anomaly_detection",
            num_class=0
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
        outputs = self.model(batch_x, None, None, None)  # torch.Size([batch_size, output_length, num_nodes])
        return outputs, batch_x


@dataclass
class PatchTSTImputation(ImputationExp, PatchTSTParameters):
    model_type: str = "PatchTST"

    def _init_model(self):
        self.model = PatchTST(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            n_heads=self.n_heads,
            dropout=self.dropout,
            e_layers=self.e_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            patch_len=self.patch_len,
            stride=self.stride,
            task_name="imputation",
            num_class=0
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
        batch_mask = batch_mask.to(self.device)
        outputs = self.model(
            batch_masked_x, None, None, None, batch_mask
        )  # torch.Size([batch_size, output_length, num_nodes])
        return outputs, batch_x


