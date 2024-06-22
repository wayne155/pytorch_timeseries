from dataclasses import dataclass, field
import sys

import torch
from ..model import Autoformer
from . import (
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
    AnomalyDetectionExp,
)


@dataclass
class AutoformerParameters:
    d_ff : int = 2048
    factor : int = 1
    activation : str = 'gelu'
    e_layers : int = 2
    d_layers : int = 1
    output_attention : bool = True
    moving_avg : list = field(default_factory=lambda : [24])
    n_heads : int = 8
    d_model : int = 512
    embed : str = 'timeF'
    dropout : float = 0.0

@dataclass
class AutoformerForecast(ForecastExp, AutoformerParameters):
    model_type: str = "Autoformer"

    def _init_model(self):
        self.label_len = int(self.windows / 2)
        
        self.model = Autoformer(
            enc_in=self.dataset.num_features,
            dec_in=self.dataset.num_features,
            seq_len=self.windows,
            pred_len=self.pred_len,
            label_len=self.label_len, 
            c_out=self.dataset.num_features,  
            factor=self.factor,
            d_ff=self.d_ff,
            activation=self.activation,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            output_attention=self.output_attention,
            moving_avg=self.moving_avg,
            n_heads=self.n_heads, 
            d_model=self.d_model,
            embed=self.embed,
            freq=self.dataset.freq,
            dropout=self.dropout, 
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
        if self.output_attention:
            outputs = self.model(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc)[0]  # torch.Size([batch_size, output_length, num_nodes])
        else:
            outputs = self.model(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc)  # torch.Size([batch_size, output_length, num_nodes])
            
        return outputs, batch_y



class AutoformerClassModel(Autoformer):
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
class AutoformerUEAClassification(UEAClassificationExp, AutoformerParameters):
    model_type: str = "Autoformer"

    def _init_model(self):
        self.label_len = self.windows/2
        self.model = AutoformerClassModel(
            enc_in=self.dataset.num_features,
            dec_in=self.dataset.num_features,
            seq_len=self.windows,
            pred_len=self.windows,
            label_len=self.label_len, 
            c_out=self.dataset.num_features,  
            factor=self.factor,
            d_ff=self.d_ff,
            activation=self.activation,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            output_attention=self.output_attention,
            moving_avg=self.moving_avg,
            n_heads=self.n_heads, 
            d_model=self.d_model,
            embed=self.embed,
            freq='h', #  not used
            dropout=self.dropout, 
            
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


class AnomalyModel(Autoformer):
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
class AutoformerAnomalyDetection(AnomalyDetectionExp, AutoformerParameters):
    model_type: str = "Autoformer"

    def _init_model(self):
        self.label_len = self.windows/2
        self.model = AnomalyModel(
            enc_in=self.dataset.num_features,
            dec_in=self.dataset.num_features,
            seq_len=self.windows,
            pred_len=self.windows,
            label_len=self.label_len, 
            c_out=self.dataset.num_features,  
            factor=self.factor,
            d_ff=self.d_ff,
            activation=self.activation,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            output_attention=self.output_attention,
            moving_avg=self.moving_avg,
            n_heads=self.n_heads, 
            d_model=self.d_model,
            embed=self.embed,
            freq='h',
            dropout=self.dropout, 
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



class ImputationModel(Autoformer):
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
class AutoformerImputation(ImputationExp, AutoformerParameters):
    model_type: str = "Autoformer"

    def _init_model(self):
        self.label_len = self.windows/2
        self.model = ImputationModel(
            enc_in=self.dataset.num_features,
            dec_in=self.dataset.num_features,
            seq_len=self.windows,
            pred_len=self.windows,
            label_len=self.label_len, 
            c_out=self.dataset.num_features,  
            factor=self.factor,
            d_ff=self.d_ff,
            activation=self.activation,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            output_attention=self.output_attention,
            moving_avg=self.moving_avg,
            n_heads=self.n_heads, 
            d_model=self.d_model,
            embed=self.embed,
            freq=self.dataset.freq,
            dropout=self.dropout, 
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


