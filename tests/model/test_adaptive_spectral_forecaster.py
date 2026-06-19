"""Tests for AdaptiveSpectralForecaster — learnable soft bandpass spectral filtering."""
import math
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.AdaptiveSpectralForecaster import AdaptiveSpectralForecaster


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=32, pred_len=8, enc_in=4, n_filters=8, revin=True,
):
    return AdaptiveSpectralForecaster(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        n_filters=n_filters, revin=revin,
    )


# ── construction ───────────────────────────────────────────────────────────────


class TestAdaptiveSpectralForecasterConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_mu_shape(self):
        m = _make_model(n_filters=8)
        assert m.mu.shape == (8,)

    def test_log_sigma_shape(self):
        m = _make_model(n_filters=8)
        assert m.log_sigma.shape == (8,)

    def test_mu_is_parameter(self):
        m = _make_model()
        assert isinstance(m.mu, nn.Parameter)

    def test_log_sigma_is_parameter(self):
        m = _make_model()
        assert isinstance(m.log_sigma, nn.Parameter)

    def test_freq_idx_is_buffer(self):
        m = _make_model(seq_len=32)
        buffer_names = {n for n, _ in m.named_buffers()}
        assert "freq_idx" in buffer_names

    def test_freq_idx_shape(self):
        m = _make_model(seq_len=32)
        n_freq = 32 // 2 + 1  # 17
        assert m.freq_idx.shape == (n_freq,)

    def test_head_shape(self):
        m = _make_model(n_filters=8, pred_len=12)
        assert m.head.in_features == 8
        assert m.head.out_features == 12

    def test_mu_initialised_uniformly(self):
        m = _make_model(seq_len=32, n_filters=4)
        # mu should span the freq range
        assert m.mu[0].item() < m.mu[-1].item()


# ── bandpass masks ─────────────────────────────────────────────────────────────


class TestBandpassMasks:
    def test_mask_shape(self):
        m = _make_model(seq_len=32, n_filters=8)
        masks = m._bandpass_masks()
        n_freq = 32 // 2 + 1
        assert masks.shape == (8, n_freq)

    def test_mask_in_0_1(self):
        m = _make_model()
        masks = m._bandpass_masks()
        assert (masks >= 0).all() and (masks <= 1).all()

    def test_mask_gradient_flows(self):
        m = _make_model()
        masks = m._bandpass_masks()
        masks.sum().backward()
        assert m.mu.grad is not None
        assert m.log_sigma.grad is not None


# ── forward ────────────────────────────────────────────────────────────────────


class TestAdaptiveSpectralForecasterForward:
    def test_output_shape(self):
        m = _make_model(seq_len=32, pred_len=8, enc_in=4)
        assert m(torch.randn(2, 32, 4)).shape == (2, 8, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        assert m(torch.randn(2, 32, 4)).shape == (2, 8, 4)

    def test_single_channel(self):
        m = AdaptiveSpectralForecaster(seq_len=16, pred_len=4, enc_in=1)
        assert m(torch.randn(2, 16, 1)).shape == (2, 4, 1)

    def test_many_channels(self):
        m = AdaptiveSpectralForecaster(seq_len=16, pred_len=4, enc_in=8)
        assert m(torch.randn(2, 16, 8)).shape == (2, 4, 8)

    def test_various_pred_lens(self):
        for pred_len in [8, 24, 96]:
            m = AdaptiveSpectralForecaster(seq_len=32, pred_len=pred_len, enc_in=4)
            assert m(torch.randn(2, 32, 4)).shape == (2, pred_len, 4)

    def test_batch_size_1(self):
        m = _make_model()
        assert m(torch.randn(1, 32, 4)).shape == (1, 8, 4)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 32, 4))).all()

    def test_gradient_flows_to_mu(self):
        m = _make_model()
        m(torch.randn(2, 32, 4)).sum().backward()
        assert m.mu.grad is not None

    def test_gradient_flows_to_log_sigma(self):
        m = _make_model()
        m(torch.randn(2, 32, 4)).sum().backward()
        assert m.log_sigma.grad is not None

    def test_gradient_flows_to_head(self):
        m = _make_model()
        m(torch.randn(2, 32, 4)).sum().backward()
        assert m.head.weight.grad is not None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 32, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))


# ── registry ───────────────────────────────────────────────────────────────────


class TestAdaptiveSpectralForecasterRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import AdaptiveSpectralForecaster as M
        assert M is AdaptiveSpectralForecaster

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.AdaptiveSpectralForecaster import AdaptiveSpectralForecasterForecast
        assert AdaptiveSpectralForecasterForecast.model_type == "AdaptiveSpectralForecaster"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            assert get_experiment_class("AdaptiveSpectralForecaster", task) is not None
