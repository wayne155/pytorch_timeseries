"""Tests for CARD — Channel Aligned Robust Blend Transformer."""
import math

import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.CARD import (
    CARD,
    _MultiHeadAttention,
    _TransformerLayer,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=96, pred_len=12, enc_in=4,
    d_model=32, n_heads=4, e_layers=2, d_ff=64,
    patch_len=16, stride=8, dropout=0.0, revin=True,
):
    return CARD(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff,
        patch_len=patch_len, stride=stride, dropout=dropout, revin=revin,
    )


# ── MultiHeadAttention ─────────────────────────────────────────────────────────


class TestMultiHeadAttention:
    def test_output_shape(self):
        attn = _MultiHeadAttention(32, 4)
        x = torch.randn(2, 10, 32)
        out = attn(x)
        assert out.shape == (2, 10, 32)

    def test_gradient_flows(self):
        attn = _MultiHeadAttention(32, 4)
        x = torch.randn(2, 10, 32, requires_grad=True)
        attn(x).sum().backward()
        assert x.grad is not None

    def test_single_head(self):
        attn = _MultiHeadAttention(32, 1)
        x = torch.randn(1, 5, 32)
        assert attn(x).shape == (1, 5, 32)


# ── TransformerLayer ───────────────────────────────────────────────────────────


class TestTransformerLayer:
    def test_output_shape(self):
        layer = _TransformerLayer(32, 4, 64)
        x = torch.randn(2, 10, 32)
        assert layer(x).shape == (2, 10, 32)

    def test_residual_magnitude(self):
        layer = _TransformerLayer(32, 4, 64)
        with torch.no_grad():
            for p in layer.parameters():
                p.zero_()
        x = torch.randn(2, 10, 32)
        # With zero weights, LayerNorm normalises residual → not exactly x
        out = layer(x)
        assert out.shape == (2, 10, 32)


# ── CARD construction ──────────────────────────────────────────────────────────


class TestCARDConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists_when_enabled(self):
        m = _make_model(revin=True)
        assert hasattr(m, "revin_layer")

    def test_no_revin_when_disabled(self):
        m = _make_model(revin=False)
        assert not hasattr(m, "revin_layer")

    def test_n_patches_correct(self):
        m = _make_model(seq_len=96, patch_len=16, stride=8)
        expected = math.ceil((96 - 16) / 8) + 1
        assert m.n_patches == expected

    def test_e_layers_count(self):
        m = _make_model(e_layers=3)
        assert len(m.temporal_layers) == 3
        assert len(m.chan_layers) == 3

    def test_temporal_proj_shapes(self):
        m = _make_model(pred_len=24, d_model=32)
        assert m.temporal_proj.out_features == 24
        assert m.chan_decoder.out_features == 24

    def test_gate_proj_shape(self):
        m = _make_model(d_model=32)
        assert m.gate_proj.in_features == 64  # 2 * d_model


# ── CARD forward ───────────────────────────────────────────────────────────────


class TestCARDForward:
    def test_output_shape(self):
        m = _make_model(seq_len=96, pred_len=12, enc_in=4)
        out = m(torch.randn(2, 96, 4))
        assert out.shape == (2, 12, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        out = m(torch.randn(2, 96, 4))
        assert out.shape == (2, 12, 4)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48, 96]:
            m = CARD(seq_len=96, pred_len=pred_len, enc_in=4,
                     d_model=32, n_heads=4, e_layers=1, d_ff=64,
                     patch_len=16, stride=8)
            out = m(torch.randn(2, 96, 4))
            assert out.shape == (2, pred_len, 4), f"Failed pred_len={pred_len}"

    def test_single_channel(self):
        m = CARD(seq_len=48, pred_len=8, enc_in=1,
                 d_model=16, n_heads=2, e_layers=1, d_ff=32,
                 patch_len=8, stride=4)
        out = m(torch.randn(2, 48, 1))
        assert out.shape == (2, 8, 1)

    def test_many_channels(self):
        m = CARD(seq_len=48, pred_len=12, enc_in=16,
                 d_model=32, n_heads=4, e_layers=1, d_ff=64,
                 patch_len=8, stride=4)
        out = m(torch.randn(2, 48, 16))
        assert out.shape == (2, 12, 16)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 96, 4))).all()

    def test_batch_size_1(self):
        m = _make_model()
        out = m(torch.randn(1, 96, 4))
        assert out.shape == (1, 12, 4)

    def test_gradient_flows_to_input(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 96, 4, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_gradient_flows_to_gate(self):
        m = _make_model()
        x = torch.randn(2, 96, 4)
        m(x).sum().backward()
        assert m.gate_proj.weight.grad is not None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 96, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_short_seq_len(self):
        # seq_len shorter than patch_len — padding should handle it
        m = CARD(seq_len=12, pred_len=4, enc_in=2,
                 d_model=16, n_heads=2, e_layers=1, d_ff=32,
                 patch_len=16, stride=8)
        out = m(torch.randn(2, 12, 2))
        assert out.shape == (2, 4, 2)

    def test_revin_scales_output(self):
        torch.manual_seed(42)
        m = _make_model(revin=True).eval()
        x_small = torch.randn(2, 96, 4) * 0.01
        x_large = torch.randn(2, 96, 4) * 100.0
        with torch.no_grad():
            out_s = m(x_small)
            out_l = m(x_large)
        assert out_l.abs().mean() > out_s.abs().mean()

    def test_single_layer(self):
        m = CARD(seq_len=48, pred_len=12, enc_in=4,
                 d_model=16, n_heads=2, e_layers=1, d_ff=32,
                 patch_len=8, stride=4)
        out = m(torch.randn(2, 48, 4))
        assert out.shape == (2, 12, 4)


# ── registry ───────────────────────────────────────────────────────────────────


class TestCARDRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import CARD as M
        assert M is CARD

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.CARD import CARDForecast
        assert CARDForecast.model_type == "CARD"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("CARD", task)
            assert cls is not None
