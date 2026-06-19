"""Tests for TSReservoir — Echo State Network forecaster."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.TSReservoir import TSReservoir


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=32, pred_len=8, enc_in=4,
    d_res=64, spectral_radius=0.9, input_scale=0.1,
    pool_states=True, revin=True,
):
    return TSReservoir(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_res=d_res, spectral_radius=spectral_radius,
        input_scale=input_scale, pool_states=pool_states, revin=revin,
    )


# ── construction ───────────────────────────────────────────────────────────────


class TestTSReservoirConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_reservoir_weights_not_parameters(self):
        m = _make_model()
        param_names = {n for n, _ in m.named_parameters()}
        assert "W_in" not in param_names
        assert "W_res" not in param_names

    def test_reservoir_weights_are_buffers(self):
        m = _make_model()
        buffer_names = {n for n, _ in m.named_buffers()}
        assert "W_in" in buffer_names
        assert "W_res" in buffer_names

    def test_w_in_shape(self):
        m = _make_model(d_res=64)
        assert m.W_in.shape == (64, 1)

    def test_w_res_shape(self):
        m = _make_model(d_res=64)
        assert m.W_res.shape == (64, 64)

    def test_spectral_radius_enforced(self):
        m = _make_model(d_res=32, spectral_radius=0.8)
        eigvals = torch.linalg.eigvals(m.W_res)
        actual_sr = eigvals.abs().max().item()
        assert abs(actual_sr - 0.8) < 0.05

    def test_readout_is_trainable(self):
        m = _make_model()
        param_names = {n for n, _ in m.named_parameters()}
        assert any("readout" in n for n in param_names)

    def test_only_readout_trained(self):
        m = _make_model()
        # Only readout and (optional) revin params
        for name, param in m.named_parameters():
            assert "readout" in name or "revin" in name, f"Unexpected trainable param: {name}"


# ── forward ────────────────────────────────────────────────────────────────────


class TestTSReservoirForward:
    def test_output_shape(self):
        m = _make_model(seq_len=32, pred_len=8, enc_in=4)
        assert m(torch.randn(2, 32, 4)).shape == (2, 8, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        assert m(torch.randn(2, 32, 4)).shape == (2, 8, 4)

    def test_output_shape_no_pool(self):
        m = _make_model(pool_states=False)
        assert m(torch.randn(2, 32, 4)).shape == (2, 8, 4)

    def test_various_pred_lens(self):
        for pred_len in [8, 16, 32]:
            m = TSReservoir(seq_len=32, pred_len=pred_len, enc_in=4, d_res=32)
            assert m(torch.randn(2, 32, 4)).shape == (2, pred_len, 4)

    def test_single_channel(self):
        m = TSReservoir(seq_len=16, pred_len=4, enc_in=1, d_res=32)
        assert m(torch.randn(2, 16, 1)).shape == (2, 4, 1)

    def test_many_channels(self):
        m = TSReservoir(seq_len=16, pred_len=4, enc_in=8, d_res=32)
        assert m(torch.randn(2, 16, 8)).shape == (2, 4, 8)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 32, 4))).all()

    def test_batch_size_1(self):
        m = _make_model()
        assert m(torch.randn(1, 32, 4)).shape == (1, 8, 4)

    def test_gradient_flows_to_readout(self):
        m = _make_model()
        m(torch.randn(2, 32, 4)).sum().backward()
        assert m.readout.weight.grad is not None

    def test_no_gradient_to_w_res(self):
        m = _make_model()
        # W_res is not a parameter so it has no .grad
        assert not hasattr(m.W_res, "grad") or m.W_res.grad is None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 32, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_revin_scales_output(self):
        torch.manual_seed(0)
        m = _make_model(revin=True).eval()
        with torch.no_grad():
            out_s = m(torch.randn(2, 32, 4) * 0.01)
            out_l = m(torch.randn(2, 32, 4) * 100.0)
        assert out_l.abs().mean() > out_s.abs().mean()


# ── registry ───────────────────────────────────────────────────────────────────


class TestTSReservoirRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import TSReservoir as M
        assert M is TSReservoir

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.TSReservoir import TSReservoirForecast
        assert TSReservoirForecast.model_type == "TSReservoir"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            assert get_experiment_class("TSReservoir", task) is not None
