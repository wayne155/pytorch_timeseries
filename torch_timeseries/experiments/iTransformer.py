from dataclasses import dataclass, field
import sys

import torch

from torch_timeseries.dataloader.sliding_window_ts import SlidingWindowTS
from torch_timeseries.utils.parse_type import parse_type
from ..model import iTransformer
from . import (
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
    AnomalyDetectionExp,
)


@dataclass
class iTransformerParameters:
    factor: int = 1
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layer: int = 1
    d_ff: int = 2048
    dropout: float = 0.1
    embed: str = "timeF"
    activation:str= "gelu"
    use_norm : bool = True
    class_strategy : str = 'projection'

@dataclass
class iTransformerForecast(ForecastExp, iTransformerParameters):
    model_type: str = "iTransformer"
        
    def _init_model(self):
        self.label_len = int(self.windows / 2)
        
        self.model = iTransformer(
            seq_len=self.windows,
            pred_len=self.pred_len,
            use_norm=self.use_norm,
            class_strategy=self.class_strategy,
            factor=self.factor,
            freq=self.dataset.freq,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            dropout=self.dropout,
            embed=self.embed,
            activation=self.activation,
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



class iTransformerClassModel(iTransformer):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

        self.projection = torch.nn.Linear(
           int( kwargs.get("d_model") * kwargs.get("c_in")), self.num_classes)

    def forward(self, x_enc):
        _, T, N = x_enc.shape # B L N
        
        enc_out = self.enc_embedding(x_enc, None) # covariates (e.g timestamp) can be also embedded as tokens
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        output = torch.relu(enc_out)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(torch.relu(output))
        return output

@dataclass
class iTransformerUEAClassification(UEAClassificationExp, iTransformerParameters):
    model_type: str = "iTransformer"

    def _init_model(self):
        self.model = iTransformerClassModel(
            seq_len=self.windows,
            pred_len=self.windows,
            use_norm=self.use_norm,
            class_strategy=self.class_strategy,
            factor=self.factor,
            freq='h',
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            dropout=self.dropout,
            embed=self.embed,
            activation=self.activation,
            num_classes=self.dataset.num_classes,
            c_in=self.dataset.num_features
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
        
        outputs = self.model(batch_x)  # torch.Size([batch_size, output_length, num_nodes])
        return outputs, batch_y.long().squeeze(-1)


# class AnomalyModel(iTransformer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.projection = torch.nn.Linear(
#             kwargs.get('d_model') , kwargs.get('c_out'))

#     def forward(self, batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc):
        
#         B,T,N = batch_x.size()
        
#         enc_out = self.enc_embedding(batch_x, None)
#         enc_out, attns = self.encoder(enc_out, attn_mask=None)
#         # final
#         dec_out = self.projection(enc_out)
#         return dec_out[:, :, :N]
    
@dataclass
class iTransformerAnomalyDetection(AnomalyDetectionExp, iTransformerParameters):
    model_type: str = "iTransformer"

    def _init_model(self):
        self.label_len = self.windows/2
        self.model = iTransformer(
            seq_len=self.windows,
            pred_len=self.windows,
            use_norm=self.use_norm,
            class_strategy=self.class_strategy,
            factor=self.factor,
            freq='h',
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            dropout=self.dropout,
            embed=self.embed,
            activation=self.activation,
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



class ImputationModel(iTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.projection = torch.nn.Linear(
            kwargs.get('d_model') , kwargs.get('pred_len'))

    def forward(self, batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc):
        enc_out = self.enc_embedding(batch_x, batch_x_date_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out
    
    
@dataclass
class iTransformerImputation(ImputationExp, iTransformerParameters):
    model_type: str = "iTransformer"

    def _init_model(self):
        self.label_len = self.windows/2
        self.model = iTransformer(
            seq_len=self.windows,
            pred_len=self.windows,
            use_norm=self.use_norm,
            class_strategy=self.class_strategy,
            factor=self.factor,
            freq=self.dataset.freq,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            dropout=self.dropout,
            embed=self.embed,
            activation=self.activation,
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


