"""Tests for torch_timeseries.nn AutoCorrelation and AutoCorrelationLayer."""
import pytest
import torch

from torch_timeseries.nn import AutoCorrelation, AutoCorrelationLayer

B, L, H, E = 4, 32, 4, 16   # batch, seq, heads, head_dim
D_MODEL = H * E              # 64


class TestAutoCorrelation:
    def _qkv(self):
        # AutoCorrelation expects (B, L, H, E)
        q = torch.randn(B, L, H, E)
        k = torch.randn(B, L, H, E)
        v = torch.randn(B, L, H, E)
        return q, k, v

    def test_output_shape_training(self):
        attn = AutoCorrelation(mask_flag=False, factor=1, attention_dropout=0.0)
        q, k, v = self._qkv()
        attn.train()
        out, _ = attn(q, k, v, attn_mask=None)
        assert out.shape == (B, L, H, E)

    def test_output_shape_inference(self):
        attn = AutoCorrelation(mask_flag=False, factor=1, attention_dropout=0.0)
        q, k, v = self._qkv()
        attn.eval()
        out, _ = attn(q, k, v, attn_mask=None)
        assert out.shape == (B, L, H, E)

    def test_no_nan_training(self):
        attn = AutoCorrelation(mask_flag=False, factor=1, attention_dropout=0.0)
        q, k, v = self._qkv()
        attn.train()
        out, _ = attn(q, k, v, attn_mask=None)
        assert not torch.isnan(out).any()

    def test_no_nan_inference(self):
        attn = AutoCorrelation(mask_flag=False, factor=1, attention_dropout=0.0)
        q, k, v = self._qkv()
        attn.eval()
        out, _ = attn(q, k, v, attn_mask=None)
        assert not torch.isnan(out).any()

    def test_output_attention(self):
        attn = AutoCorrelation(mask_flag=False, factor=1, attention_dropout=0.0,
                                output_attention=True)
        q, k, v = self._qkv()
        attn.eval()
        out, corr = attn(q, k, v, attn_mask=None)
        assert out.shape == (B, L, H, E)
        assert corr is not None

    def test_cross_attention_shorter_keys(self):
        S = 16
        attn = AutoCorrelation(mask_flag=False, factor=1, attention_dropout=0.0)
        q = torch.randn(B, L, H, E)
        k = torch.randn(B, S, H, E)
        v = torch.randn(B, S, H, E)
        attn.eval()
        out, _ = attn(q, k, v, attn_mask=None)
        assert out.shape == (B, L, H, E)

    def test_gradients_flow(self):
        attn = AutoCorrelation(mask_flag=False, factor=1, attention_dropout=0.0)
        q = torch.randn(B, L, H, E, requires_grad=True)
        k = torch.randn(B, L, H, E)
        v = torch.randn(B, L, H, E)
        attn.train()
        out, _ = attn(q, k, v, attn_mask=None)
        out.sum().backward()
        assert q.grad is not None


class TestAutoCorrelationLayer:
    def test_output_shape(self):
        layer = AutoCorrelationLayer(
            AutoCorrelation(mask_flag=False, factor=1, attention_dropout=0.0),
            d_model=D_MODEL, n_heads=H,
        )
        x = torch.randn(B, L, D_MODEL)
        layer.train()
        out, _ = layer(x, x, x, attn_mask=None)
        assert out.shape == (B, L, D_MODEL)

    def test_output_shape_inference(self):
        layer = AutoCorrelationLayer(
            AutoCorrelation(mask_flag=False, factor=1, attention_dropout=0.0),
            d_model=D_MODEL, n_heads=H,
        )
        x = torch.randn(B, L, D_MODEL)
        layer.eval()
        out, _ = layer(x, x, x, attn_mask=None)
        assert out.shape == (B, L, D_MODEL)

    def test_no_nan(self):
        layer = AutoCorrelationLayer(
            AutoCorrelation(mask_flag=False, factor=1, attention_dropout=0.0),
            d_model=D_MODEL, n_heads=H,
        )
        x = torch.randn(B, L, D_MODEL)
        layer.train()
        out, _ = layer(x, x, x, attn_mask=None)
        assert not torch.isnan(out).any()

    def test_gradients_flow(self):
        layer = AutoCorrelationLayer(
            AutoCorrelation(mask_flag=False, factor=1, attention_dropout=0.0),
            d_model=D_MODEL, n_heads=H,
        )
        x = torch.randn(B, L, D_MODEL, requires_grad=True)
        layer.train()
        out, _ = layer(x, x, x, attn_mask=None)
        out.sum().backward()
        assert x.grad is not None

    def test_cross_attention_shape(self):
        S = 24
        layer = AutoCorrelationLayer(
            AutoCorrelation(mask_flag=False, factor=1, attention_dropout=0.0),
            d_model=D_MODEL, n_heads=H,
        )
        q = torch.randn(B, L, D_MODEL)
        kv = torch.randn(B, S, D_MODEL)
        layer.eval()
        out, _ = layer(q, kv, kv, attn_mask=None)
        assert out.shape == (B, L, D_MODEL)
