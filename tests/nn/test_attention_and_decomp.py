"""Tests for torch_timeseries.nn attention and decomposition modules."""
import pytest
import torch

from torch_timeseries.nn import (
    AttentionLayer,
    FullAttention,
    MovingAvg,
    SeriesDecomp,
    SeriesDecompMulti,
)

B, L, H, E = 4, 32, 4, 16   # batch, seq, heads, head_dim
D_MODEL = H * E              # 64


# ── FullAttention ─────────────────────────────────────────────────────────────

class TestFullAttention:
    def _qkv(self):
        q = torch.randn(B, L, H, E)
        k = torch.randn(B, L, H, E)
        v = torch.randn(B, L, H, E)
        return q, k, v

    def test_output_shape(self):
        attn = FullAttention(mask_flag=False, attention_dropout=0.0)
        q, k, v = self._qkv()
        out, scores = attn(q, k, v)
        assert out.shape == (B, L, H, E)

    def test_output_shape_cross(self):
        S = 48
        attn = FullAttention(mask_flag=False, attention_dropout=0.0)
        q = torch.randn(B, L, H, E)
        k = torch.randn(B, S, H, E)
        v = torch.randn(B, S, H, E)
        out, _ = attn(q, k, v)
        assert out.shape == (B, L, H, E)

    def test_masked_attention(self):
        attn = FullAttention(mask_flag=True, attention_dropout=0.0)
        q, k, v = self._qkv()
        out, _ = attn(q, k, v)
        assert out.shape == (B, L, H, E)
        assert not torch.isnan(out).any()

    def test_no_nan_no_mask(self):
        attn = FullAttention(mask_flag=False, attention_dropout=0.0)
        q, k, v = self._qkv()
        out, _ = attn(q, k, v)
        assert not torch.isnan(out).any()

    def test_gradients_flow(self):
        attn = FullAttention(mask_flag=False, attention_dropout=0.0)
        q = torch.randn(B, L, H, E, requires_grad=True)
        k = torch.randn(B, L, H, E)
        v = torch.randn(B, L, H, E)
        out, _ = attn(q, k, v)
        out.sum().backward()
        assert q.grad is not None

    def test_custom_scale(self):
        attn = FullAttention(mask_flag=False, scale=0.1, attention_dropout=0.0)
        q, k, v = self._qkv()
        out, _ = attn(q, k, v)
        assert out.shape == (B, L, H, E)


# ── AttentionLayer ────────────────────────────────────────────────────────────

class TestAttentionLayer:
    def test_output_shape(self):
        layer = AttentionLayer(
            FullAttention(mask_flag=False), d_model=D_MODEL, n_heads=H
        )
        x = torch.randn(B, L, D_MODEL)
        out, attn = layer(x, x, x, attn_mask=None)
        assert out.shape == (B, L, D_MODEL)

    def test_cross_attention_shape(self):
        S = 48
        layer = AttentionLayer(
            FullAttention(mask_flag=False), d_model=D_MODEL, n_heads=H
        )
        q = torch.randn(B, L, D_MODEL)
        kv = torch.randn(B, S, D_MODEL)
        out, _ = layer(q, kv, kv, attn_mask=None)
        assert out.shape == (B, L, D_MODEL)

    def test_gradients_flow(self):
        layer = AttentionLayer(
            FullAttention(mask_flag=False), d_model=D_MODEL, n_heads=H
        )
        x = torch.randn(B, L, D_MODEL, requires_grad=True)
        out, _ = layer(x, x, x, attn_mask=None)
        out.sum().backward()
        assert x.grad is not None

    def test_no_nan(self):
        layer = AttentionLayer(
            FullAttention(mask_flag=False), d_model=D_MODEL, n_heads=H
        )
        x = torch.randn(B, L, D_MODEL)
        out, _ = layer(x, x, x, attn_mask=None)
        assert not torch.isnan(out).any()


# ── MovingAvg ─────────────────────────────────────────────────────────────────

class TestMovingAvg:
    def test_output_shape_odd_kernel(self):
        ma = MovingAvg(kernel_size=5, stride=1)
        x = torch.randn(B, L, 7)
        out = ma(x)
        assert out.shape == (B, L, 7)

    def test_constant_series_unchanged(self):
        ma = MovingAvg(kernel_size=5, stride=1)
        x = torch.ones(1, 32, 3) * 2.0
        out = ma(x)
        assert torch.allclose(out, x, atol=1e-5)

    def test_no_nan(self):
        ma = MovingAvg(kernel_size=7, stride=1)
        x = torch.randn(B, L, 7)
        assert not torch.isnan(ma(x)).any()


# ── SeriesDecomp ──────────────────────────────────────────────────────────────

class TestSeriesDecomp:
    def test_output_shape(self):
        sd = SeriesDecomp(kernel_size=25)
        x = torch.randn(B, L, 7)
        residual, trend = sd(x)
        assert residual.shape == (B, L, 7)
        assert trend.shape == (B, L, 7)

    def test_decomp_adds_to_original(self):
        sd = SeriesDecomp(kernel_size=25)
        x = torch.randn(B, L, 7)
        residual, trend = sd(x)
        assert torch.allclose(residual + trend, x, atol=1e-5)

    def test_trend_smoother_than_residual(self):
        sd = SeriesDecomp(kernel_size=25)
        x = torch.randn(1, 100, 1)
        residual, trend = sd(x)
        # Trend should have lower variance than the original
        assert trend.var() <= x.var()

    def test_no_nan(self):
        sd = SeriesDecomp(kernel_size=5)
        x = torch.randn(B, L, 7)
        res, trend = sd(x)
        assert not torch.isnan(res).any()
        assert not torch.isnan(trend).any()


# ── SeriesDecompMulti ─────────────────────────────────────────────────────────

class TestSeriesDecompMulti:
    def test_output_shape(self):
        sdm = SeriesDecompMulti(kernel_size=[13, 17, 25])
        x = torch.randn(B, L, 7)
        residual, trend = sdm(x)
        assert residual.shape == (B, L, 7)
        assert trend.shape == (B, L, 7)

    def test_decomp_sums_to_original(self):
        sdm = SeriesDecompMulti(kernel_size=[13, 17])
        x = torch.randn(B, L, 7)
        residual, trend = sdm(x)
        assert torch.allclose(residual + trend, x, atol=1e-4)

    def test_gradients_flow(self):
        sdm = SeriesDecompMulti(kernel_size=[13, 25])
        x = torch.randn(B, L, 7, requires_grad=True)
        res, trend = sdm(x)
        (res + trend).sum().backward()
        assert x.grad is not None

    def test_no_nan(self):
        sdm = SeriesDecompMulti(kernel_size=[5, 7, 11])
        x = torch.randn(B, L, 7)
        res, trend = sdm(x)
        assert not torch.isnan(res).any()
        assert not torch.isnan(trend).any()

    def test_single_kernel(self):
        sdm = SeriesDecompMulti(kernel_size=[25])
        x = torch.randn(B, L, 7)
        res, trend = sdm(x)
        assert res.shape == trend.shape == (B, L, 7)
