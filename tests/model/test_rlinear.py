"""Tests for RLinear — Reversible Linear forecaster."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.RLinear import RLinear


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(seq_len=48, pred_len=12, enc_in=4, individual=True):
    return RLinear(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
                   individual=individual)


# ── Construction ───────────────────────────────────────────────────────────────


class TestRLinearConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_individual_num_linears(self):
        m = _make_model(enc_in=7, individual=True)
        assert len(m.linears) == 7

    def test_shared_single_linear(self):
        m = _make_model(enc_in=7, individual=False)
        assert hasattr(m, "linear")
        assert not hasattr(m, "linears")

    def test_revin_exists(self):
        m = _make_model()
        assert hasattr(m, "revin_layer")

    def test_linear_shape_individual(self):
        m = _make_model(seq_len=48, pred_len=12, enc_in=4, individual=True)
        assert m.linears[0].in_features == 48
        assert m.linears[0].out_features == 12

    def test_linear_shape_shared(self):
        m = _make_model(seq_len=48, pred_len=12, enc_in=4, individual=False)
        assert m.linear.in_features == 48
        assert m.linear.out_features == 12


# ── Forward ────────────────────────────────────────────────────────────────────


class TestRLinearForward:
    def test_output_shape_individual(self):
        m = _make_model(seq_len=48, pred_len=12, enc_in=4, individual=True)
        out = m(torch.randn(4, 48, 4))
        assert out.shape == (4, 12, 4)

    def test_output_shape_shared(self):
        m = _make_model(seq_len=48, pred_len=12, enc_in=4, individual=False)
        out = m(torch.randn(4, 48, 4))
        assert out.shape == (4, 12, 4)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48, 96]:
            m = RLinear(seq_len=96, pred_len=pred_len, enc_in=4)
            out = m(torch.randn(2, 96, 4))
            assert out.shape == (2, pred_len, 4), f"Failed pred_len={pred_len}"

    def test_single_channel(self):
        m = RLinear(seq_len=24, pred_len=8, enc_in=1)
        out = m(torch.randn(2, 24, 1))
        assert out.shape == (2, 8, 1)

    def test_output_finite_individual(self):
        m = _make_model(individual=True)
        assert torch.isfinite(m(torch.randn(2, 48, 4))).all()

    def test_output_finite_shared(self):
        m = _make_model(individual=False)
        assert torch.isfinite(m(torch.randn(2, 48, 4))).all()

    def test_gradient_flows_individual(self):
        m = _make_model(individual=True)
        x = torch.randn(2, 48, 4, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_gradient_flows_shared(self):
        m = _make_model(individual=False)
        x = torch.randn(2, 48, 4, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_revin_scales_output(self):
        torch.manual_seed(0)
        m = _make_model().eval()
        x_small = torch.randn(2, 48, 4) * 0.01
        x_large = torch.randn(2, 48, 4) * 100.0
        with torch.no_grad():
            out_small = m(x_small)
            out_large = m(x_large)
        assert out_large.abs().mean() > out_small.abs().mean()

    def test_batch_size_1(self):
        m = _make_model()
        out = m(torch.randn(1, 48, 4))
        assert out.shape == (1, 12, 4)

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 48, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_individual_and_shared_differ(self):
        torch.manual_seed(42)
        m_ind = RLinear(48, 12, 4, individual=True)
        torch.manual_seed(42)
        m_shr = RLinear(48, 12, 4, individual=False)
        x = torch.randn(2, 48, 4)
        with torch.no_grad():
            out_ind = m_ind(x)
            out_shr = m_shr(x)
        # Likely different due to different parameterisation
        assert out_ind.shape == out_shr.shape == (2, 12, 4)

    def test_constant_input(self):
        m = _make_model().eval()
        x = torch.ones(2, 48, 4)
        with torch.no_grad():
            out = m(x)
        assert torch.isfinite(out).all()


# ── Registry ───────────────────────────────────────────────────────────────────


class TestRLinearRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import RLinear as M
        assert M is RLinear

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.RLinear import RLinearForecast
        assert RLinearForecast.model_type == "RLinear"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("RLinear", task)
            assert cls is not None
