"""Tests for torch_timeseries.nn Encoder/Decoder Transformer building blocks."""
import pytest
import torch
import torch.nn as nn

from torch_timeseries.nn import (
    AttentionLayer,
    ConvLayer,
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    EncoderStack,
    FullAttention,
)

B, L, S, D = 4, 48, 24, 64
N_HEADS = 4


def _attn_layer(mask_flag=False):
    return AttentionLayer(
        FullAttention(mask_flag=mask_flag, attention_dropout=0.0),
        d_model=D, n_heads=N_HEADS,
    )


# ── ConvLayer ─────────────────────────────────────────────────────────────────

class TestConvLayer:
    def test_output_shape_halves_length(self):
        conv = ConvLayer(c_in=D)
        x = torch.randn(B, L, D)
        out = conv(x)
        # MaxPool1d(kernel_size=3, stride=2, padding=1) halves the length
        assert out.shape[0] == B
        assert out.shape[2] == D
        assert out.shape[1] < L

    def test_no_nan(self):
        conv = ConvLayer(c_in=D)
        assert not torch.isnan(conv(torch.randn(B, L, D))).any()

    def test_gradients_flow(self):
        conv = ConvLayer(c_in=D)
        x = torch.randn(B, L, D, requires_grad=True)
        conv(x).sum().backward()
        assert x.grad is not None


# ── EncoderLayer ──────────────────────────────────────────────────────────────

class TestEncoderLayer:
    def test_output_shape(self):
        layer = EncoderLayer(_attn_layer(), d_model=D, d_ff=D * 4, dropout=0.0)
        x = torch.randn(B, L, D)
        out, attn = layer(x)
        assert out.shape == (B, L, D)

    def test_no_nan(self):
        layer = EncoderLayer(_attn_layer(), d_model=D, dropout=0.0)
        out, _ = layer(torch.randn(B, L, D))
        assert not torch.isnan(out).any()

    def test_gradients_flow(self):
        layer = EncoderLayer(_attn_layer(), d_model=D, dropout=0.0)
        x = torch.randn(B, L, D, requires_grad=True)
        out, _ = layer(x)
        out.sum().backward()
        assert x.grad is not None

    def test_gelu_activation(self):
        layer = EncoderLayer(_attn_layer(), d_model=D, dropout=0.0, activation="gelu")
        out, _ = layer(torch.randn(B, L, D))
        assert out.shape == (B, L, D)


# ── Encoder ───────────────────────────────────────────────────────────────────

class TestEncoder:
    def test_single_layer(self):
        enc = Encoder(
            attn_layers=[EncoderLayer(_attn_layer(), d_model=D, dropout=0.0)],
            norm_layer=nn.LayerNorm(D),
        )
        x = torch.randn(B, L, D)
        out, attns = enc(x)
        assert out.shape == (B, L, D)
        assert len(attns) == 1

    def test_two_layers(self):
        enc = Encoder(
            attn_layers=[
                EncoderLayer(_attn_layer(), d_model=D, dropout=0.0),
                EncoderLayer(_attn_layer(), d_model=D, dropout=0.0),
            ],
        )
        out, attns = enc(torch.randn(B, L, D))
        assert out.shape == (B, L, D)
        assert len(attns) == 2

    def test_with_conv_layers(self):
        enc = Encoder(
            attn_layers=[
                EncoderLayer(_attn_layer(), d_model=D, dropout=0.0),
                EncoderLayer(_attn_layer(), d_model=D, dropout=0.0),
            ],
            conv_layers=[ConvLayer(c_in=D)],
        )
        out, attns = enc(torch.randn(B, L, D))
        assert out.shape[2] == D
        assert not torch.isnan(out).any()

    def test_no_nan(self):
        enc = Encoder([EncoderLayer(_attn_layer(), d_model=D, dropout=0.0)])
        out, _ = enc(torch.randn(B, L, D))
        assert not torch.isnan(out).any()

    def test_gradients_flow(self):
        enc = Encoder([EncoderLayer(_attn_layer(), d_model=D, dropout=0.0)])
        x = torch.randn(B, L, D, requires_grad=True)
        out, _ = enc(x)
        out.sum().backward()
        assert x.grad is not None


# ── DecoderLayer ──────────────────────────────────────────────────────────────

class TestDecoderLayer:
    def test_output_shape(self):
        layer = DecoderLayer(
            self_attention=_attn_layer(),
            cross_attention=_attn_layer(),
            d_model=D, dropout=0.0,
        )
        x = torch.randn(B, S, D)
        memory = torch.randn(B, L, D)
        out = layer(x, memory)
        assert out.shape == (B, S, D)

    def test_no_nan(self):
        layer = DecoderLayer(_attn_layer(), _attn_layer(), d_model=D, dropout=0.0)
        out = layer(torch.randn(B, S, D), torch.randn(B, L, D))
        assert not torch.isnan(out).any()

    def test_gradients_flow(self):
        layer = DecoderLayer(_attn_layer(), _attn_layer(), d_model=D, dropout=0.0)
        x = torch.randn(B, S, D, requires_grad=True)
        memory = torch.randn(B, L, D)
        layer(x, memory).sum().backward()
        assert x.grad is not None


# ── Decoder ───────────────────────────────────────────────────────────────────

class TestDecoder:
    def test_output_shape(self):
        dec = Decoder(
            layers=[
                DecoderLayer(_attn_layer(), _attn_layer(), d_model=D, dropout=0.0),
            ],
            norm_layer=nn.LayerNorm(D),
        )
        x = torch.randn(B, S, D)
        memory = torch.randn(B, L, D)
        out = dec(x, memory)
        assert out.shape == (B, S, D)

    def test_with_norm(self):
        dec = Decoder(
            layers=[DecoderLayer(_attn_layer(), _attn_layer(), d_model=D, dropout=0.0)],
            norm_layer=nn.LayerNorm(D),
        )
        out = dec(torch.randn(B, S, D), torch.randn(B, L, D))
        assert out.shape == (B, S, D)

    def test_no_nan(self):
        dec = Decoder([DecoderLayer(_attn_layer(), _attn_layer(), d_model=D, dropout=0.0)])
        out = dec(torch.randn(B, S, D), torch.randn(B, L, D))
        assert not torch.isnan(out).any()

    def test_two_layers(self):
        dec = Decoder([
            DecoderLayer(_attn_layer(), _attn_layer(), d_model=D, dropout=0.0),
            DecoderLayer(_attn_layer(), _attn_layer(), d_model=D, dropout=0.0),
        ])
        out = dec(torch.randn(B, S, D), torch.randn(B, L, D))
        assert out.shape == (B, S, D)
