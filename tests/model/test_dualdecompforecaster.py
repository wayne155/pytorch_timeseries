"""Tests for DualDecompForecaster — trend/seasonal decomposition with dual branches."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.DualDecompForecaster import (
    DualDecompForecaster, _MovingAvgDecomp,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=64, pred_len=12, enc_in=4,
    kernel_size=7, d_model=32, n_heads=4, e_layers=2, d_ff=64,
    patch_len=8, dropout=0.0, revin=True,
):
    return DualDecompForecaster(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        kernel_size=kernel_size, d_model=d_model, n_heads=n_heads,
        e_layers=e_layers, d_ff=d_ff, patch_len=patch_len,
        dropout=dropout, revin=revin,
    )


# ── MovingAvgDecomp unit tests ─────────────────────────────────────────────────


class TestMovingAvgDecomp:
    def test_output_lengths_match_input(self):
        d = _MovingAvgDecomp(kernel_size=5)
        x = torch.randn(4, 32)
        trend, seasonal = d(x)
        assert trend.shape == x.shape
        assert seasonal.shape == x.shape

    def test_trend_plus_seasonal_equals_input(self):
        d = _MovingAvgDecomp(kernel_size=5)
        x = torch.randn(4, 32)
        trend, seasonal = d(x)
        assert torch.allclose(trend + seasonal, x, atol=1e-5)

    def test_trend_smoother_than_input(self):
        d = _MovingAvgDecomp(kernel_size=7)
        x = torch.randn(4, 64)
        trend, _ = d(x)
        # Trend variance should be lower than input variance
        assert trend.var() < x.var()

    def test_gradient_flows(self):
        d = _MovingAvgDecomp(kernel_size=5)
        x = torch.randn(4, 32, requires_grad=True)
        trend, seasonal = d(x)
        (trend + seasonal).sum().backward()
        assert x.grad is not None


# ── DualDecompForecaster construction ─────────────────────────────────────────


class TestDualDecompForecasterConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_trend_head_dimensions(self):
        m = _make_model(seq_len=64, pred_len=12)
        assert m.trend_head.in_features == 64
        assert m.trend_head.out_features == 12

    def test_seasonal_layer_count(self):
        m = _make_model(e_layers=3)
        assert len(m.seasonal_layers) == 3

    def test_kernel_clamped_to_odd(self):
        m = DualDecompForecaster(seq_len=64, pred_len=12, enc_in=4, kernel_size=8)
        # kernel_size=8 → bumped to 9 (odd)
        assert m.decomp.kernel_size % 2 == 1


# ── DualDecompForecaster forward ──────────────────────────────────────────────


class TestDualDecompForecasterForward:
    def test_output_shape(self):
        m = _make_model(seq_len=64, pred_len=12, enc_in=4)
        assert m(torch.randn(2, 64, 4)).shape == (2, 12, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        assert m(torch.randn(2, 64, 4)).shape == (2, 12, 4)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48]:
            m = DualDecompForecaster(seq_len=96, pred_len=pred_len, enc_in=4,
                                     kernel_size=9, d_model=16, n_heads=2,
                                     e_layers=1, d_ff=32, patch_len=8)
            assert m(torch.randn(2, 96, 4)).shape == (2, pred_len, 4)

    def test_single_channel(self):
        m = DualDecompForecaster(seq_len=32, pred_len=8, enc_in=1,
                                  kernel_size=5, d_model=16, n_heads=2,
                                  e_layers=1, d_ff=32, patch_len=8)
        assert m(torch.randn(2, 32, 1)).shape == (2, 8, 1)

    def test_many_channels(self):
        m = DualDecompForecaster(seq_len=32, pred_len=8, enc_in=16,
                                  kernel_size=5, d_model=16, n_heads=2,
                                  e_layers=1, d_ff=32, patch_len=8)
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

    def test_gradient_flows_to_trend_head(self):
        m = _make_model()
        m(torch.randn(2, 64, 4)).sum().backward()
        assert m.trend_head.weight.grad is not None

    def test_gradient_flows_to_seasonal_layers(self):
        m = _make_model()
        m(torch.randn(2, 64, 4)).sum().backward()
        assert m.seasonal_layers[0].ff1.weight.grad is not None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 64, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_large_kernel_clamped(self):
        m = DualDecompForecaster(seq_len=32, pred_len=8, enc_in=4, kernel_size=100,
                                  d_model=16, n_heads=2, e_layers=1, d_ff=32, patch_len=8)
        assert m(torch.randn(2, 32, 4)).shape == (2, 8, 4)

    def test_revin_scales_output(self):
        torch.manual_seed(0)
        m = _make_model(revin=True).eval()
        with torch.no_grad():
            out_s = m(torch.randn(2, 64, 4) * 0.01)
            out_l = m(torch.randn(2, 64, 4) * 100.0)
        assert out_l.abs().mean() > out_s.abs().mean()


# ── registry ───────────────────────────────────────────────────────────────────


class TestDualDecompForecasterRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import DualDecompForecaster as M
        assert M is DualDecompForecaster

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.DualDecompForecaster import DualDecompForecasterForecast
        assert DualDecompForecasterForecast.model_type == "DualDecompForecaster"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            assert get_experiment_class("DualDecompForecaster", task) is not None
