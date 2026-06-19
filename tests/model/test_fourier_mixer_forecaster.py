"""Tests for FourierMixerForecaster — DFT frequency-domain MLP mixer."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.FourierMixerForecaster import FourierMixerForecaster, _FreqMixLayer


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=32, pred_len=8, enc_in=4,
    e_layers=2, dropout=0.1, revin=True,
):
    return FourierMixerForecaster(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        e_layers=e_layers, dropout=dropout, revin=revin,
    )


# ── freq mix layer ─────────────────────────────────────────────────────────────


class TestFreqMixLayer:
    def test_output_shape(self):
        layer = _FreqMixLayer(seq_len=32)
        x = torch.randn(4, 32)
        assert layer(x).shape == (4, 32)

    def test_mix_r_shape(self):
        layer = _FreqMixLayer(seq_len=32)
        n_freq = 32 // 2 + 1  # 17
        assert layer.mix_r.in_features == n_freq
        assert layer.mix_r.out_features == n_freq

    def test_mix_i_shape(self):
        layer = _FreqMixLayer(seq_len=32)
        n_freq = 32 // 2 + 1
        assert layer.mix_i.in_features == n_freq
        assert layer.mix_i.out_features == n_freq

    def test_output_is_real(self):
        layer = _FreqMixLayer(seq_len=16)
        out = layer(torch.randn(2, 16))
        assert out.is_floating_point()
        assert not out.is_complex()


# ── construction ───────────────────────────────────────────────────────────────


class TestFourierMixerForecasterConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_num_layers(self):
        m = _make_model(e_layers=3)
        assert len(m.layers) == 3

    def test_head_shape(self):
        m = _make_model(seq_len=32, pred_len=12)
        assert m.head.in_features == 32
        assert m.head.out_features == 12


# ── forward ────────────────────────────────────────────────────────────────────


class TestFourierMixerForecasterForward:
    def test_output_shape(self):
        m = _make_model(seq_len=32, pred_len=8, enc_in=4)
        assert m(torch.randn(2, 32, 4)).shape == (2, 8, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        assert m(torch.randn(2, 32, 4)).shape == (2, 8, 4)

    def test_single_channel(self):
        m = FourierMixerForecaster(seq_len=16, pred_len=4, enc_in=1)
        assert m(torch.randn(2, 16, 1)).shape == (2, 4, 1)

    def test_many_channels(self):
        m = FourierMixerForecaster(seq_len=16, pred_len=4, enc_in=8)
        assert m(torch.randn(2, 16, 8)).shape == (2, 4, 8)

    def test_various_pred_lens(self):
        for pred_len in [8, 24, 96]:
            m = FourierMixerForecaster(seq_len=32, pred_len=pred_len, enc_in=4)
            assert m(torch.randn(2, 32, 4)).shape == (2, pred_len, 4)

    def test_batch_size_1(self):
        m = _make_model()
        assert m(torch.randn(1, 32, 4)).shape == (1, 8, 4)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 32, 4))).all()

    def test_gradient_flows(self):
        m = _make_model()
        m(torch.randn(2, 32, 4)).sum().backward()
        assert m.layers[0].mix_r.weight.grad is not None
        assert m.layers[0].mix_i.weight.grad is not None
        assert m.head.weight.grad is not None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 32, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_output_is_real(self):
        m = _make_model()
        out = m(torch.randn(2, 32, 4))
        assert not out.is_complex()


# ── registry ───────────────────────────────────────────────────────────────────


class TestFourierMixerForecasterRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import FourierMixerForecaster as M
        assert M is FourierMixerForecaster

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.FourierMixerForecaster import FourierMixerForecasterForecast
        assert FourierMixerForecasterForecast.model_type == "FourierMixerForecaster"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            assert get_experiment_class("FourierMixerForecaster", task) is not None
