"""Tests for WaveletForecaster — multi-resolution Haar wavelet decomposition."""
import math
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.WaveletForecaster import WaveletForecaster, _HaarDWT1d


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=64, pred_len=12, enc_in=4,
    n_levels=3, revin=True,
):
    return WaveletForecaster(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        n_levels=n_levels, revin=revin,
    )


# ── HaarDWT unit tests ─────────────────────────────────────────────────────────


class TestHaarDWT1d:
    def test_output_lengths_even(self):
        dwt = _HaarDWT1d()
        x = torch.randn(4, 64)
        a, d = dwt(x)
        assert a.shape == (4, 32)
        assert d.shape == (4, 32)

    def test_output_lengths_odd(self):
        dwt = _HaarDWT1d()
        x = torch.randn(4, 63)
        a, d = dwt(x)
        assert a.shape == (4, 32)
        assert d.shape == (4, 32)

    def test_energy_approximately_preserved(self):
        dwt = _HaarDWT1d()
        torch.manual_seed(0)
        x = torch.randn(4, 64)
        a, d = dwt(x)
        # Parseval for Haar: ‖x‖² ≈ ‖a‖² + ‖d‖²
        energy_in = x.pow(2).sum()
        energy_out = a.pow(2).sum() + d.pow(2).sum()
        assert abs(energy_in.item() - energy_out.item()) < 1e-3

    def test_haar_filters_not_parameters(self):
        dwt = _HaarDWT1d()
        assert len(list(dwt.parameters())) == 0

    def test_gradient_flows_through_dwt(self):
        dwt = _HaarDWT1d()
        x = torch.randn(4, 32, requires_grad=True)
        a, d = dwt(x)
        (a.sum() + d.sum()).backward()
        assert x.grad is not None


# ── WaveletForecaster construction ─────────────────────────────────────────────


class TestWaveletForecasterConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_band_head_count(self):
        # n_levels details + 1 approx = n_levels + 1 heads
        m = _make_model(n_levels=3)
        assert len(m.band_heads) == 4

    def test_band_head_sizes_correct(self):
        m = WaveletForecaster(seq_len=64, pred_len=12, enc_in=4, n_levels=3)
        # T=64: level 1 → 32, level 2 → 16, level 3 → 8; approx → 8
        expected_in = [32, 16, 8, 8]
        actual_in = [h.in_features for h in m.band_heads]
        assert actual_in == expected_in

    def test_band_head_out_features(self):
        m = _make_model(pred_len=24)
        for h in m.band_heads:
            assert h.out_features == 24

    def test_n_levels_1(self):
        m = WaveletForecaster(seq_len=32, pred_len=8, enc_in=4, n_levels=1)
        assert len(m.band_heads) == 2


# ── WaveletForecaster forward ──────────────────────────────────────────────────


class TestWaveletForecasterForward:
    def test_output_shape(self):
        m = _make_model(seq_len=64, pred_len=12, enc_in=4)
        assert m(torch.randn(2, 64, 4)).shape == (2, 12, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        assert m(torch.randn(2, 64, 4)).shape == (2, 12, 4)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48, 96]:
            m = WaveletForecaster(seq_len=96, pred_len=pred_len, enc_in=4, n_levels=3)
            assert m(torch.randn(2, 96, 4)).shape == (2, pred_len, 4)

    def test_odd_seq_len(self):
        m = WaveletForecaster(seq_len=63, pred_len=12, enc_in=4, n_levels=2)
        assert m(torch.randn(2, 63, 4)).shape == (2, 12, 4)

    def test_single_channel(self):
        m = WaveletForecaster(seq_len=32, pred_len=8, enc_in=1, n_levels=2)
        assert m(torch.randn(2, 32, 1)).shape == (2, 8, 1)

    def test_many_channels(self):
        m = WaveletForecaster(seq_len=32, pred_len=8, enc_in=16, n_levels=2)
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

    def test_gradient_flows_to_band_heads(self):
        m = _make_model()
        m(torch.randn(2, 64, 4)).sum().backward()
        for head in m.band_heads:
            assert head.weight.grad is not None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 64, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_single_level(self):
        m = WaveletForecaster(seq_len=32, pred_len=8, enc_in=4, n_levels=1)
        assert m(torch.randn(2, 32, 4)).shape == (2, 8, 4)

    def test_revin_scales_output(self):
        torch.manual_seed(0)
        m = _make_model(revin=True).eval()
        with torch.no_grad():
            out_s = m(torch.randn(2, 64, 4) * 0.01)
            out_l = m(torch.randn(2, 64, 4) * 100.0)
        assert out_l.abs().mean() > out_s.abs().mean()


# ── registry ───────────────────────────────────────────────────────────────────


class TestWaveletForecasterRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import WaveletForecaster as M
        assert M is WaveletForecaster

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.WaveletForecaster import WaveletForecasterForecast
        assert WaveletForecasterForecast.model_type == "WaveletForecaster"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            assert get_experiment_class("WaveletForecaster", task) is not None
