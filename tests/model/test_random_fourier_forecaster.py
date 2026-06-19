"""Tests for RandomFourierForecaster — RFF kernel approximation forecaster."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.RandomFourierForecaster import RandomFourierForecaster


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=32, pred_len=8, enc_in=4,
    d_rff=64, sigma=1.0, revin=True,
):
    return RandomFourierForecaster(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_rff=d_rff, sigma=sigma, revin=revin,
    )


# ── construction ───────────────────────────────────────────────────────────────


class TestRandomFourierForecasterConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_W_is_buffer_not_param(self):
        m = _make_model()
        param_names = {n for n, _ in m.named_parameters()}
        assert "W" not in param_names

    def test_b_is_buffer_not_param(self):
        m = _make_model()
        param_names = {n for n, _ in m.named_parameters()}
        assert "b" not in param_names

    def test_W_shape(self):
        m = _make_model(seq_len=32, d_rff=64)
        assert m.W.shape == (64, 32)

    def test_b_shape(self):
        m = _make_model(d_rff=64)
        assert m.b.shape == (64,)

    def test_readout_shape(self):
        m = _make_model(d_rff=64, pred_len=12)
        assert m.readout.in_features == 64
        assert m.readout.out_features == 12

    def test_only_readout_and_revin_trained(self):
        m = _make_model()
        for name, _ in m.named_parameters():
            assert "readout" in name or "revin" in name, f"Unexpected trainable: {name}"


# ── forward ────────────────────────────────────────────────────────────────────


class TestRandomFourierForecasterForward:
    def test_output_shape(self):
        m = _make_model(seq_len=32, pred_len=8, enc_in=4)
        assert m(torch.randn(2, 32, 4)).shape == (2, 8, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        assert m(torch.randn(2, 32, 4)).shape == (2, 8, 4)

    def test_single_channel(self):
        m = RandomFourierForecaster(seq_len=16, pred_len=4, enc_in=1)
        assert m(torch.randn(2, 16, 1)).shape == (2, 4, 1)

    def test_many_channels(self):
        m = RandomFourierForecaster(seq_len=16, pred_len=4, enc_in=8)
        assert m(torch.randn(2, 16, 8)).shape == (2, 4, 8)

    def test_various_pred_lens(self):
        for pred_len in [8, 24, 96]:
            m = RandomFourierForecaster(seq_len=32, pred_len=pred_len, enc_in=4, d_rff=32)
            assert m(torch.randn(2, 32, 4)).shape == (2, pred_len, 4)

    def test_batch_size_1(self):
        m = _make_model()
        assert m(torch.randn(1, 32, 4)).shape == (1, 8, 4)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 32, 4))).all()

    def test_gradient_flows_to_readout(self):
        m = _make_model()
        m(torch.randn(2, 32, 4)).sum().backward()
        assert m.readout.weight.grad is not None

    def test_no_gradient_to_W(self):
        m = _make_model()
        m(torch.randn(2, 32, 4)).sum().backward()
        # W is a buffer, not a parameter — no .grad attribute that matters
        assert not isinstance(m.W, nn.Parameter)

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 32, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_rff_cosine_range(self):
        """RFF features should lie in [-√(2/D), √(2/D)] × D range."""
        import math
        m = _make_model(d_rff=64).eval()
        x_ci = torch.randn(4, 32)
        with torch.no_grad():
            proj = x_ci @ m.W.T + m.b
            phi = math.sqrt(2.0 / 64) * torch.cos(proj)
        # cos is bounded [-1,1], scaling by sqrt(2/D) keeps values small
        assert phi.abs().max().item() <= math.sqrt(2.0 / 64) + 1e-5


# ── registry ───────────────────────────────────────────────────────────────────


class TestRandomFourierForecasterRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import RandomFourierForecaster as M
        assert M is RandomFourierForecaster

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.RandomFourierForecaster import RandomFourierForecasterForecast
        assert RandomFourierForecasterForecast.model_type == "RandomFourierForecaster"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            assert get_experiment_class("RandomFourierForecaster", task) is not None
