import torch
import torch.nn as nn
import torch.nn.functional as F
from ..nn.Transformer_EncDec import Encoder, EncoderLayer
from ..nn.SelfAttention_Family import FullAttention, AttentionLayer
from ..nn.embedding import DataEmbedding_inverted
import numpy as np


class iTransformer(nn.Module):
    """iTransformer — Inverted Transformer for multivariate forecasting (Liu et al., ICLR 2024).

    Inverts the roles of the time and feature axes: each *variable* (rather
    than each time step) is treated as a token, so attention captures
    multivariate correlations while the feed-forward network encodes
    temporal patterns within each variable.

    Paper: *iTransformer: Inverted Transformers Are Effective for Time Series
    Forecasting.*
    https://openreview.net/forum?id=JePfAI8fah

    Args:
        seq_len (int): Input sequence length.
        pred_len (int): Prediction horizon length.
        c_in (int): Number of input features (channels). Defaults to 7.
        d_model (int): Embedding dimension. Defaults to 512.
        n_heads (int): Number of attention heads. Defaults to 8.
        e_layers (int): Number of encoder layers. Defaults to 2.
        d_ff (int): Feed-forward hidden size. Defaults to 2048.
        dropout (float): Dropout probability. Defaults to 0.1.
        use_norm (bool): Apply reversible instance normalisation. Defaults to True.
        class_strategy (str): Projection strategy for multi-task heads.
            Defaults to ``'projection'``.

    Tasks: Forecasting, Imputation, Anomaly Detection, Classification.
    """

    def __init__(self, seq_len, pred_len, use_norm=True,c_in=7, factor=1,n_heads=8, d_ff=2048, activation='gelu', e_layers=2, output_attention=True, d_model=512, embed='timeF', freq='h', dropout=0.1, class_strategy='projection'):
        super(iTransformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.use_norm = use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq,
                                                    dropout)
        self.class_strategy = class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]