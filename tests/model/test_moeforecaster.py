"""Tests for MoEForecaster — Mixture-of-Experts time-series forecaster."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.MoEForecaster import (
    MoEForecaster,
    _LinearExpert,
    _MLPExpert,
    _Router,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=48, pred_len=12, enc_in=4,
    n_experts=4, k_active=2, d_router=16,
    expert_type="linear", d_ff=64,
    dropout=0.0, revin=True,
):
    return MoEForecaster(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        n_experts=n_experts, k_active=k_active, d_router=d_router,
        expert_type=expert_type, d_ff=d_ff, dropout=dropout, revin=revin,
    )


# ── LinearExpert ───────────────────────────────────────────────────────────────


class TestLinearExpert:
    def test_output_shape(self):
        exp = _LinearExpert(seq_len=48, pred_len=12, dropout=0.0)
        assert exp(torch.randn(8, 48)).shape == (8, 12)

    def test_gradient_flows(self):
        exp = _LinearExpert(seq_len=24, pred_len=8, dropout=0.0)
        x = torch.randn(4, 24, requires_grad=True)
        exp(x).sum().backward()
        assert x.grad is not None


# ── MLPExpert ──────────────────────────────────────────────────────────────────


class TestMLPExpert:
    def test_output_shape(self):
        exp = _MLPExpert(seq_len=48, pred_len=12, d_ff=32, dropout=0.0)
        assert exp(torch.randn(8, 48)).shape == (8, 12)

    def test_gradient_flows(self):
        exp = _MLPExpert(seq_len=24, pred_len=8, d_ff=32, dropout=0.0)
        x = torch.randn(4, 24, requires_grad=True)
        exp(x).sum().backward()
        assert x.grad is not None


# ── Router ─────────────────────────────────────────────────────────────────────


class TestRouter:
    def test_output_shape(self):
        router = _Router(n_experts=6, d_router=16, k_active=2, dropout=0.0)
        x = torch.randn(8, 48)
        w = router(x)
        assert w.shape == (8, 6)

    def test_weights_sum_to_one(self):
        router = _Router(n_experts=4, d_router=16, k_active=2, dropout=0.0)
        w = router(torch.randn(8, 48))
        assert torch.allclose(w.sum(-1), torch.ones(8), atol=1e-5)

    def test_sparse_k_active(self):
        router = _Router(n_experts=6, d_router=16, k_active=2, dropout=0.0)
        w = router(torch.randn(4, 48))
        # Exactly n_experts - k_active entries should be zero (or near zero)
        nonzero_per_row = (w > 1e-6).sum(-1)
        assert (nonzero_per_row <= 2).all()

    def test_k_active_ge_n_experts_is_dense(self):
        router = _Router(n_experts=3, d_router=16, k_active=3, dropout=0.0)
        w = router(torch.randn(4, 48))
        assert (w > 0).all()

    def test_stats_gradient_flows(self):
        router = _Router(n_experts=4, d_router=16, k_active=2, dropout=0.0)
        x = torch.randn(4, 24, requires_grad=True)
        router(x).sum().backward()
        assert x.grad is not None


# ── MoEForecaster construction ─────────────────────────────────────────────────


class TestMoEForecasterConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_expert_count(self):
        m = _make_model(n_experts=6)
        assert len(m.experts) == 6

    def test_mlp_expert_type(self):
        m = _make_model(expert_type="mlp")
        assert isinstance(m.experts[0], _MLPExpert)

    def test_linear_expert_type(self):
        m = _make_model(expert_type="linear")
        assert isinstance(m.experts[0], _LinearExpert)


# ── MoEForecaster forward ──────────────────────────────────────────────────────


class TestMoEForecasterForward:
    def test_output_shape(self):
        m = _make_model(seq_len=48, pred_len=12, enc_in=4)
        assert m(torch.randn(2, 48, 4)).shape == (2, 12, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        assert m(torch.randn(2, 48, 4)).shape == (2, 12, 4)

    def test_mlp_experts(self):
        m = _make_model(expert_type="mlp", d_ff=32)
        assert m(torch.randn(2, 48, 4)).shape == (2, 12, 4)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48]:
            m = MoEForecaster(seq_len=48, pred_len=pred_len, enc_in=4,
                              n_experts=4, k_active=2)
            assert m(torch.randn(2, 48, 4)).shape == (2, pred_len, 4)

    def test_single_channel(self):
        m = MoEForecaster(seq_len=24, pred_len=8, enc_in=1, n_experts=4, k_active=2)
        assert m(torch.randn(2, 24, 1)).shape == (2, 8, 1)

    def test_many_channels(self):
        m = MoEForecaster(seq_len=24, pred_len=8, enc_in=16, n_experts=4, k_active=2)
        assert m(torch.randn(2, 24, 16)).shape == (2, 8, 16)

    def test_k_active_1(self):
        m = MoEForecaster(seq_len=48, pred_len=12, enc_in=4, n_experts=4, k_active=1)
        assert m(torch.randn(2, 48, 4)).shape == (2, 12, 4)

    def test_k_active_equals_n_experts(self):
        m = MoEForecaster(seq_len=48, pred_len=12, enc_in=4, n_experts=4, k_active=4)
        assert m(torch.randn(2, 48, 4)).shape == (2, 12, 4)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 48, 4))).all()

    def test_gradient_flows_to_input(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 48, 4, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_gradient_flows_to_experts(self):
        m = _make_model()
        m(torch.randn(2, 48, 4)).sum().backward()
        assert m.experts[0].linear.weight.grad is not None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 48, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_revin_scales_output(self):
        torch.manual_seed(0)
        m = _make_model(revin=True).eval()
        with torch.no_grad():
            assert m(torch.randn(2, 48, 4) * 100.0).abs().mean() > \
                   m(torch.randn(2, 48, 4) * 0.01).abs().mean()


# ── registry ───────────────────────────────────────────────────────────────────


class TestMoEForecasterRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import MoEForecaster as M
        assert M is MoEForecaster

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.MoEForecaster import MoEForecasterForecast
        assert MoEForecasterForecast.model_type == "MoEForecaster"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            assert get_experiment_class("MoEForecaster", task) is not None
