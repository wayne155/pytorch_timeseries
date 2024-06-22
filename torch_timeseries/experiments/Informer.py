from dataclasses import dataclass, field
import sys

import torch

from torch_timeseries.dataloader.sliding_window_ts import SlidingWindowTS
from torch_timeseries.utils.parse_type import parse_type
from ..model import Informer
from . import (
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
    AnomalyDetectionExp,
)


@dataclass
class InformerParameters:
    factor: int = 5
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layer: int = 512
    d_ff: int = 512
    dropout: float = 0.0
    attn: str = "prob"
    embed: str = "timeF"
    activation = "gelu"
    distil: bool = True
    mix: bool = True

@dataclass
class InformerForecast(ForecastExp, InformerParameters):
    model_type: str = "Informer"
        
    def _init_model(self):
        self.label_len = int(self.windows / 2)
        
        self.model = Informer(
            self.dataset.num_features,
            self.dataset.num_features,
            self.dataset.num_features,
            self.pred_len,
            factor=self.factor,
            freq=self.dataset.freq,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            dropout=self.dropout,
            attn=self.attn,
            embed=self.embed,
            activation=self.activation,
            distil=self.distil,
            mix=self.mix,
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



class InformerClassModel(Informer):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.t_projection = torch.nn.Linear(
           int(  self.out_len/2), self.out_len)

        self.projection = torch.nn.Linear(
           int( kwargs.get("d_model") * self.out_len), self.num_classes)

    def forward(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = self.t_projection(enc_out.transpose(1,2)).transpose(1,2)
        
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = torch.relu(enc_out)
        # # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(torch.relu(output))
        return output

@dataclass
class InformerUEAClassification(UEAClassificationExp, InformerParameters):
    model_type: str = "Informer"

    def _init_model(self):
        self.label_len = self.windows/2
        self.model = InformerClassModel(
            enc_in=self.dataset.num_features,
            dec_in=self.dataset.num_features,
            c_out=self.dataset.num_features,
            out_len=self.windows,
            factor=self.factor,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            dropout=self.dropout,
            attn=self.attn,
            embed=self.embed,
            activation=self.activation,
            distil=self.distil,
            mix=self.mix,
            
            num_classes=self.dataset.num_classes
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
        
        outputs = self.model(batch_x, padding_masks)  # torch.Size([batch_size, output_length, num_nodes])
        return outputs, batch_y.long().squeeze(-1)


class AnomalyModel(Informer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.projection = torch.nn.Linear(
            kwargs.get('d_model') , kwargs.get('c_out'))
        self.t_projection = torch.nn.Linear(
        int(self.out_len/2) ,self.out_len)

    def forward(self, batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc):
        enc_out = self.enc_embedding(batch_x, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        dec_out = self.t_projection(dec_out.transpose(1,2)).transpose(1,2)
        return dec_out
    
@dataclass
class InformerAnomalyDetection(AnomalyDetectionExp, InformerParameters):
    model_type: str = "Informer"

    def _init_model(self):
        self.label_len = self.windows/2
        self.model = AnomalyModel(
            enc_in=self.dataset.num_features,
            dec_in=self.dataset.num_features,
            c_out=self.dataset.num_features,
            out_len=self.windows,
            factor=self.factor,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            dropout=self.dropout,
            attn=self.attn,
            embed=self.embed,
            activation=self.activation,
            distil=self.distil,
            mix=self.mix,
        )
        self.model = self.model.to(self.device)


    def _process_one_batch(self, batch_x, origin_x, batch_y):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(batch_x, None, None, None)  # torch.Size([batch_size, output_length, num_nodes])
        return outputs, batch_x



class ImputationModel(Informer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.projection = torch.nn.Linear(
            kwargs.get('d_model') , kwargs.get('c_out'))
        self.t_projection = torch.nn.Linear(
        int(self.out_len/2) ,self.out_len)

    def forward(self, batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc):
        enc_out = self.enc_embedding(batch_x, batch_x_date_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        dec_out = self.t_projection(dec_out.transpose(1,2)).transpose(1,2)
        return dec_out
    
    
@dataclass
class InformerImputation(ImputationExp, InformerParameters):
    model_type: str = "Informer"

    def _init_model(self):
        self.label_len = self.windows/2
        self.model = ImputationModel(
            enc_in=self.dataset.num_features,
            dec_in=self.dataset.num_features,
            c_out=self.dataset.num_features,
            out_len=self.windows,
            factor=self.factor,
            freq=self.dataset.freq,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            dropout=self.dropout,
            attn=self.attn,
            embed=self.embed,
            activation=self.activation,
            distil=self.distil,
            mix=self.mix,
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
        batch_x_date_enc = batch_x_date_enc.to(self.device, dtype=torch.float32)
        batch_x = batch_x.to(self.device).float()
        
        outputs = self.model(
            batch_masked_x, batch_x_date_enc, None, None
        )  # torch.Size([batch_size, output_length, num_nodes])
        
        
        return outputs, batch_x


