"""Tests for FilterNet — learnable frequency-domain filter bank."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.FilterNet import (
    FilterNet,
    _FrequencyFilter,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(seq_len=48, pred_len=12, enc_in=4, num_filters=4, revin=True):
    return FilterNet(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        num_filters=num_filters, revin=revin,
    )


# ── FrequencyFilter ────────────────────────────────────────────────────────────


class TestFrequencyFilter:
    def test_output_shape(self):
        n_freq = 25
        filt = _FrequencyFilter(n_freq)
        X = torch.randn(8, n_freq, dtype=torch.complex64)
        out = filt(X)
        assert out.shape == (8, n_freq)

    def test_weights_in_unit_interval(self):
        filt = _FrequencyFilter(25)
        w = torch.sigmoid(filt.weights)
        assert (w > 0).all() and (w < 1).all()

    def test_gradient_flows(self):
        filt = _FrequencyFilter(25)
        X = torch.randn(4, 25, dtype=torch.complex64)
        out = filt(X)
        out.real.sum().backward()
        assert filt.weights.grad is not None

    def test_zero_weight_zeros_output(self):
        """Setting all weights → 0 (via large negative log-odds) should zero spectrum."""
        filt = _FrequencyFilter(25)
        with torch.no_grad():
            filt.weights.fill_(-100.0)
        X = torch.randn(4, 25, dtype=torch.complex64)
        out = filt(X)
        assert out.abs().max() < 1e-3


# ── FilterNet construction ─────────────────────────────────────────────────────


class TestFilterNetConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_num_filters(self):
        m = _make_model(num_filters=6)
        assert len(m.filters) == 6

    def test_revin_exists_when_enabled(self):
        m = _make_model(revin=True)
        assert hasattr(m, "revin_layer")

    def test_no_revin_when_disabled(self):
        m = _make_model(revin=False)
        assert not hasattr(m, "revin_layer")

    def test_decoder_shape(self):
        m = _make_model(seq_len=48, pred_len=12)
        assert m.decoder.in_features == 48
        assert m.decoder.out_features == 12

    def test_n_freq_correct(self):
        m = _make_model(seq_len=48)
        assert m.n_freq == 48 // 2 + 1


# ── FilterNet forward ──────────────────────────────────────────────────────────


class TestFilterNetForward:
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
            m = FilterNet(seq_len=96, pred_len=pred_len, enc_in=4, num_filters=4)
            out = m(torch.randn(2, 96, 4))
            assert out.shape == (2, pred_len, 4), f"Failed pred_len={pred_len}"

    def test_single_channel(self):
        m = FilterNet(seq_len=24, pred_len=8, enc_in=1, num_filters=3)
        out = m(torch.randn(2, 24, 1))
        assert out.shape == (2, 8, 1)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 48, 4))).all()

    def test_gradient_flows_to_input(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 48, 4, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_gradient_flows_to_filter_weights(self):
        m = _make_model()
        x = torch.randn(2, 48, 4)
        m(x).sum().backward()
        for filt in m.filters:
            assert filt.weights.grad is not None

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

    def test_num_filters_1(self):
        m = FilterNet(seq_len=48, pred_len=12, enc_in=4, num_filters=1)
        out = m(torch.randn(2, 48, 4))
        assert out.shape == (2, 12, 4)


# ── registry ───────────────────────────────────────────────────────────────────


class TestFilterNetRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import FilterNet as M
        assert M is FilterNet

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.FilterNet import FilterNetForecast
        assert FilterNetForecast.model_type == "FilterNet"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("FilterNet", task)
            assert cls is not None
