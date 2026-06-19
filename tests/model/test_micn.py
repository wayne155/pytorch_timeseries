"""Tests for MICN — Multi-scale Isometric Convolution Network."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.MICN import (
    MICN,
    _IsometricConvBlock,
    _SeasonalPredictionBlock,
    _TrendPredictionBlock,
)


# ── helpers ──────────────────────────��────────────────────────���────────────────


def _make_model(seq_len=48, pred_len=12, enc_in=4, d_model=32, num_scales=2,
                kernel_size=3, dropout=0.0, revin=True):
    return MICN(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_model=d_model, num_scales=num_scales, kernel_size=kernel_size,
        dropout=dropout, revin=revin,
    )


# ── IsometricConvBlock ────────────────────────────────────────────────��────────


class TestIsometricConvBlock:
    def test_output_shape_preserved(self):
        block = _IsometricConvBlock(enc_in=4, d_model=32, kernel_size=5, dilation=1, dropout=0.0)
        x = torch.randn(2, 48, 4)
        out = block(x)
        assert out.shape == (2, 48, 4), "conv block must preserve sequence length"

    def test_output_finite(self):
        block = _IsometricConvBlock(enc_in=4, d_model=32, kernel_size=5, dilation=1, dropout=0.0)
        x = torch.randn(2, 48, 4)
        assert torch.isfinite(block(x)).all()

    def test_dilation_2_preserves_length(self):
        block = _IsometricConvBlock(enc_in=4, d_model=16, kernel_size=3, dilation=2, dropout=0.0)
        x = torch.randn(2, 48, 4)
        assert block(x).shape == (2, 48, 4)

    def test_gradient_flows(self):
        block = _IsometricConvBlock(enc_in=4, d_model=16, kernel_size=3, dilation=1, dropout=0.0)
        x = torch.randn(2, 24, 4, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None


# ── SeasonalPredictionBlock ───────────────────────��────────────────────────────


class TestSeasonalPredictionBlock:
    def test_output_shape(self):
        spb = _SeasonalPredictionBlock(
            seq_len=48, pred_len=12, enc_in=4, d_model=32, num_scales=3,
            kernel_size=5, dropout=0.0,
        )
        x = torch.randn(2, 48, 4)
        out = spb(x)
        assert out.shape == (2, 12, 4)

    def test_num_branches(self):
        spb = _SeasonalPredictionBlock(
            seq_len=48, pred_len=12, enc_in=4, d_model=32, num_scales=4,
            kernel_size=3, dropout=0.0,
        )
        assert len(spb.branches) == 4

    def test_gradient_flows(self):
        spb = _SeasonalPredictionBlock(48, 12, 4, 32, 2, 3, 0.0)
        x = torch.randn(2, 48, 4, requires_grad=True)
        spb(x).sum().backward()
        assert x.grad is not None


# ── TrendPredictionBlock ───────────────────────────────────────────────────────


class TestTrendPredictionBlock:
    def test_output_shape(self):
        tpb = _TrendPredictionBlock(seq_len=48, pred_len=12, enc_in=4)
        x = torch.randn(2, 48, 4)
        out = tpb(x)
        assert out.shape == (2, 12, 4)

    def test_gradient_flows(self):
        tpb = _TrendPredictionBlock(seq_len=48, pred_len=12, enc_in=4)
        x = torch.randn(2, 48, 4, requires_grad=True)
        tpb(x).sum().backward()
        assert x.grad is not None


# ── MICN construction ──────────────────────────────────────────────────────────


class TestMICNConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists_when_enabled(self):
        m = _make_model(revin=True)
        assert hasattr(m, "revin_layer")

    def test_no_revin_when_disabled(self):
        m = _make_model(revin=False)
        assert not hasattr(m, "revin_layer")

    def test_spb_has_correct_num_branches(self):
        m = _make_model(num_scales=4)
        assert len(m.spb.branches) == 4


# ── MICN forward ─────────────────────���────────────────────────────────────────


class TestMICNForward:
    def test_output_shape(self):
        m = _make_model(seq_len=48, pred_len=12, enc_in=4)
        out = m(torch.randn(4, 48, 4))
        assert out.shape == (4, 12, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        out = m(torch.randn(4, 48, 4))
        assert out.shape == (4, 12, 4)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48, 96]:
            m = MICN(seq_len=96, pred_len=pred_len, enc_in=4, d_model=32, num_scales=2,
                     kernel_size=3)
            out = m(torch.randn(2, 96, 4))
            assert out.shape == (2, pred_len, 4), f"Failed for pred_len={pred_len}"

    def test_single_channel(self):
        m = MICN(seq_len=24, pred_len=8, enc_in=1, d_model=16, num_scales=2, kernel_size=3)
        out = m(torch.randn(3, 24, 1))
        assert out.shape == (3, 8, 1)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 48, 4))).all()

    def test_gradient_flows(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 48, 4, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_batch_size_1(self):
        m = _make_model()
        out = m(torch.randn(1, 48, 4))
        assert out.shape == (1, 12, 4)

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 48, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_revin_scales_output(self):
        torch.manual_seed(0)
        m = _make_model(revin=True).eval()
        x_small = torch.randn(2, 48, 4) * 0.01
        x_large = torch.randn(2, 48, 4) * 100.0
        with torch.no_grad():
            out_small = m(x_small)
            out_large = m(x_large)
        assert out_large.abs().mean() > out_small.abs().mean()

    def test_num_scales_1_works(self):
        m = MICN(seq_len=48, pred_len=12, enc_in=4, d_model=16, num_scales=1, kernel_size=3)
        out = m(torch.randn(2, 48, 4))
        assert out.shape == (2, 12, 4)


# ── registry ───────────────────────────���───────────────────────────────────────


class TestMICNRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import MICN as M
        assert M is MICN

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.MICN import MICNForecast
        assert MICNForecast.model_type == "MICN"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("MICN", task)
            assert cls is not None
