from dataclasses import dataclass, field
import sys

import torch
from ..model import FEDformer
from . import (
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
    AnomalyDetectionExp,
)
# ,  version='Fourier', L=3, d_ff=2048, activation='gelu', e_layers=2, d_layers=1, mode_select='random', modes=64, output_attention=True, moving_avg=[24] , n_heads=8, cross_activation='tanh', d_model=512, embed='timeF', freq='h', dropout=0.0, base='legendre'

@dataclass
class FEDformerParameters:
    d_ff : float = 2048
    d_model : int = 512
    embed : str = 'timeF'
    dropout : float = 0.0
    cross_activation : str = 'tanh'
    activation : str = 'gelu'
    version : str = 'Fourier'
    n_heads : int = 8
    L : int = 3
    moving_avg : list = field(default_factory=lambda : [24])
    e_layers  : int = 2
    d_layers  : int = 1
    modes  : int = 64
    base : str = 'legendre'
    mode_select : str = 'random'
@dataclass
class FEDformerForecast(ForecastExp, FEDformerParameters):
    model_type: str = "FEDformer"

    def _init_model(self):
        self.label_len = int(self.windows / 2)
        
        self.model = FEDformer(
            enc_in=self.dataset.num_features,
            dec_in=self.dataset.num_features,
            seq_len=self.windows,
            pred_len=self.pred_len,
            label_len=self.label_len, 
            c_out=self.dataset.num_features,  
            version=self.version, 
            L=self.L, 
            d_ff=self.d_ff, 
            activation=self.activation,
            e_layers=self.e_layers, 
            d_layers=self.d_layers,
            mode_select=self.mode_select,
            modes=self.modes,
            output_attention=True,
            moving_avg=self.moving_avg,
            n_heads=self.n_heads,
            cross_activation=self.cross_activation,
            d_model=self.d_model,
            embed=self.embed,
            freq=self.dataset.freq,
            dropout=self.dropout,
            base=self.base
            
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
        outputs = self.model(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc)[0]  # torch.Size([batch_size, output_length, num_nodes])
        
        # if self.output_attention:
        #     outputs = self.model(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc)[0]  # torch.Size([batch_size, output_length, num_nodes])
        # else:
        #     outputs = self.model(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc)  # torch.Size([batch_size, output_length, num_nodes])
            
        return outputs, batch_y



class FEDformerClassModel(FEDformer):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.projection = torch.nn.Linear(
            self.d_model * self.seq_len, self.num_classes)

    def forward(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = torch.relu(enc_out)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(torch.relu(output))
        return output

@dataclass
class FEDformerUEAClassification(UEAClassificationExp, FEDformerParameters):
    model_type: str = "FEDformer"

    def _init_model(self):
        self.label_len = self.windows/2
        self.model = FEDformerClassModel(
            enc_in=self.dataset.num_features,
            dec_in=self.dataset.num_features,
            seq_len=self.windows,
            pred_len=self.windows,
            label_len=self.label_len, 
            c_out=self.dataset.num_features,  
            version=self.version, 
            L=self.L, 
            d_ff=self.d_ff, 
            activation=self.activation,
            e_layers=self.e_layers, 
            d_layers=self.d_layers,
            mode_select=self.mode_select,
            modes=self.modes,
            output_attention=True,
            moving_avg=self.moving_avg,
            n_heads=self.n_heads,
            cross_activation=self.cross_activation,
            d_model=self.d_model,
            embed=self.embed,
            freq='h',
            dropout=self.dropout,
            base=self.base,

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


class AnomalyModel(FEDformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.projection = torch.nn.Linear(
            self.d_model , self.c_out)
    
    def forward(self, batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc):
        enc_out = self.enc_embedding(batch_x, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

@dataclass
class FEDformerAnomalyDetection(AnomalyDetectionExp, FEDformerParameters):
    model_type: str = "FEDformer"

    def _init_model(self):
        self.label_len = self.windows/2
        self.model = AnomalyModel(
            enc_in=self.dataset.num_features,
            dec_in=self.dataset.num_features,
            seq_len=self.windows,
            pred_len=self.windows,
            label_len=self.label_len, 
            c_out=self.dataset.num_features,  
            version=self.version, 
            L=self.L, 
            d_ff=self.d_ff, 
            activation=self.activation,
            e_layers=self.e_layers, 
            d_layers=self.d_layers,
            mode_select=self.mode_select,
            modes=self.modes,
            output_attention=True,
            moving_avg=self.moving_avg,
            n_heads=self.n_heads,
            cross_activation=self.cross_activation,
            d_model=self.d_model,
            embed=self.embed,
            freq='h',
            dropout=self.dropout,
            base=self.base
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



class ImputationModel(FEDformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.projection = torch.nn.Linear(
            self.d_model , self.c_out)
    
    def forward(self, batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc):
        enc_out = self.enc_embedding(batch_x, batch_x_date_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out
@dataclass
class FEDformerImputation(ImputationExp, FEDformerParameters):
    model_type: str = "FEDformer"

    def _init_model(self):
        self.label_len = self.windows/2
        self.model = ImputationModel(
            enc_in=self.dataset.num_features,
            dec_in=self.dataset.num_features,
            seq_len=self.windows,
            pred_len=self.windows,
            label_len=self.label_len, 
            c_out=self.dataset.num_features,  
            version=self.version, 
            L=self.L, 
            d_ff=self.d_ff, 
            activation=self.activation,
            e_layers=self.e_layers, 
            d_layers=self.d_layers,
            mode_select=self.mode_select,
            modes=self.modes,
            output_attention=True,
            moving_avg=self.moving_avg,
            n_heads=self.n_heads,
            cross_activation=self.cross_activation,
            d_model=self.d_model,
            embed=self.embed,
            freq=self.dataset.freq,
            dropout=self.dropout,
            base=self.base
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


