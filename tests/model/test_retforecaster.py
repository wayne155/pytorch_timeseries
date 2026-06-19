"""Tests for RetForecaster — Retentive Network for time-series forecasting."""
import math

import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.RetForecaster import (
    RetForecaster,
    _MultiScaleRetention,
    _RetNetLayer,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=64, pred_len=12, enc_in=4,
    d_model=32, n_heads=4, e_layers=2, d_ff=64,
    patch_len=8, stride=8, dropout=0.0, revin=True,
):
    return RetForecaster(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff,
        patch_len=patch_len, stride=stride, dropout=dropout, revin=revin,
    )


# ── MultiScaleRetention ────────────────────────────────────────────────────────


class TestMultiScaleRetention:
    def test_output_shape(self):
        msr = _MultiScaleRetention(d_model=32, n_heads=4)
        x = torch.randn(2, 10, 32)
        assert msr(x).shape == (2, 10, 32)

    def test_gradient_flows(self):
        msr = _MultiScaleRetention(d_model=16, n_heads=2)
        x = torch.randn(2, 5, 16, requires_grad=True)
        msr(x).sum().backward()
        assert x.grad is not None

    def test_gamma_in_0_1(self):
        msr = _MultiScaleRetention(d_model=32, n_heads=4)
        gamma = torch.sigmoid(msr.gamma)
        assert (gamma > 0).all() and (gamma < 1).all()

    def test_causal_mask_lower_triangular(self):
        msr = _MultiScaleRetention(d_model=16, n_heads=2)
        gamma = torch.sigmoid(msr.gamma)
        D = msr._causal_decay_mask(5, gamma)
        # Upper triangle (above diagonal) should be zero
        assert (D[:, 0, 1:] == 0).all()
        assert (D[:, 0, 0] > 0).all()


# ── RetNetLayer ────────────────────────────────────────────────────────────────


class TestRetNetLayer:
    def test_output_shape(self):
        layer = _RetNetLayer(d_model=32, n_heads=4, d_ff=64)
        x = torch.randn(2, 8, 32)
        assert layer(x).shape == (2, 8, 32)

    def test_gradient_flows(self):
        layer = _RetNetLayer(d_model=16, n_heads=2, d_ff=32)
        x = torch.randn(2, 5, 16, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None


# ── RetForecaster construction ─────────────────────────────────────────────────


class TestRetForecasterConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_n_patches_correct(self):
        m = _make_model(seq_len=64, patch_len=8, stride=8)
        assert m.n_patches == math.ceil((64 - 8) / 8) + 1

    def test_layer_count(self):
        m = _make_model(e_layers=3)
        assert len(m.layers) == 3

    def test_head_out_features(self):
        m = _make_model(pred_len=24)
        assert m.head.out_features == 24


# ── RetForecaster forward ──────────────────────────────────────────────────────


class TestRetForecasterForward:
    def test_output_shape(self):
        m = _make_model(seq_len=64, pred_len=12, enc_in=4)
        out = m(torch.randn(2, 64, 4))
        assert out.shape == (2, 12, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        assert m(torch.randn(2, 64, 4)).shape == (2, 12, 4)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48]:
            m = RetForecaster(seq_len=96, pred_len=pred_len, enc_in=4,
                              d_model=16, n_heads=2, e_layers=1, d_ff=32,
                              patch_len=16, stride=16)
            assert m(torch.randn(2, 96, 4)).shape == (2, pred_len, 4)

    def test_single_channel(self):
        m = RetForecaster(seq_len=32, pred_len=8, enc_in=1,
                          d_model=16, n_heads=2, e_layers=1, d_ff=32,
                          patch_len=8, stride=8)
        assert m(torch.randn(2, 32, 1)).shape == (2, 8, 1)

    def test_many_channels(self):
        m = RetForecaster(seq_len=32, pred_len=8, enc_in=16,
                          d_model=16, n_heads=2, e_layers=1, d_ff=32,
                          patch_len=8, stride=8)
        assert m(torch.randn(2, 32, 16)).shape == (2, 8, 16)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 64, 4))).all()

    def test_batch_size_1(self):
        m = _make_model()
        assert m(torch.randn(1, 64, 4)).shape == (1, 12, 4)

    def test_gradient_flows_to_input(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 64, 4, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_gradient_flows_to_gamma(self):
        m = _make_model()
        m(torch.randn(2, 64, 4)).sum().backward()
        assert m.layers[0].msr.gamma.grad is not None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 64, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_revin_scales_output(self):
        torch.manual_seed(0)
        m = _make_model(revin=True).eval()
        with torch.no_grad():
            out_s = m(torch.randn(2, 64, 4) * 0.01)
            out_l = m(torch.randn(2, 64, 4) * 100.0)
        assert out_l.abs().mean() > out_s.abs().mean()


# ── registry ───────────────────────────────────────────────────────────────────


class TestRetForecasterRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import RetForecaster as M
        assert M is RetForecaster

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.RetForecaster import RetForecasterForecast
        assert RetForecasterForecast.model_type == "RetForecaster"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            assert get_experiment_class("RetForecaster", task) is not None
