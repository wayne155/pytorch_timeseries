"""Tests for SparseTSF — period-based sparse forecasting model."""
import math
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.SparseTSF import SparseTSF


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_model(seq_len=96, pred_len=24, enc_in=7, period=None, revin=True):
    return SparseTSF(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, period=period, revin=revin
    )


# ── construction ──────────────────────────────────────────────────────────────


class TestSparseTSFConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_default_period_is_quarter_seq_len(self):
        m = SparseTSF(seq_len=96, pred_len=24, enc_in=7)
        assert m.period == 24   # 96 // 4

    def test_custom_period_stored(self):
        m = SparseTSF(seq_len=96, pred_len=24, enc_in=7, period=12)
        assert m.period == 12

    def test_period_1_is_identity_downsampling(self):
        # period=1: no downsampling, pred_steps = pred_len
        m = SparseTSF(seq_len=24, pred_len=12, enc_in=3, period=1)
        assert m.T_down == 24
        assert m.pred_steps == 12

    def test_revin_layer_exists_when_enabled(self):
        m = SparseTSF(seq_len=96, pred_len=24, enc_in=7, revin=True)
        assert hasattr(m, "revin_layer")

    def test_no_revin_layer_when_disabled(self):
        m = SparseTSF(seq_len=96, pred_len=24, enc_in=7, revin=False)
        assert not hasattr(m, "revin_layer")

    def test_few_parameters(self):
        """SparseTSF should have far fewer params than attention-based models."""
        m = SparseTSF(seq_len=720, pred_len=96, enc_in=7, period=24, revin=False)
        n_params = sum(p.numel() for p in m.parameters())
        # T_down=30, pred_steps=4 → Linear: 30*4+4=124 params (excluding RevIN)
        assert n_params < 500

    def test_tiny_parameter_count_with_revin(self):
        m = SparseTSF(seq_len=720, pred_len=96, enc_in=7, period=24, revin=True)
        n_params = sum(p.numel() for p in m.parameters())
        # ~124 + 14 = 138 parameters — still tiny
        assert n_params < 1000


# ── forward ───────────────────────────────────────────────────────────────────


class TestSparseTSFForward:
    def test_output_shape(self):
        m = _make_model(seq_len=96, pred_len=24, enc_in=7)
        out = m(torch.randn(4, 96, 7))
        assert out.shape == (4, 24, 7)

    def test_output_shape_no_revin(self):
        m = _make_model(seq_len=96, pred_len=24, enc_in=7, revin=False)
        out = m(torch.randn(4, 96, 7))
        assert out.shape == (4, 24, 7)

    def test_output_shape_pred_equals_seq(self):
        m = _make_model(seq_len=48, pred_len=48, enc_in=3)
        out = m(torch.randn(2, 48, 3))
        assert out.shape == (2, 48, 3)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48, 96]:
            m = SparseTSF(seq_len=96, pred_len=pred_len, enc_in=7, period=24)
            out = m(torch.randn(2, 96, 7))
            assert out.shape == (2, pred_len, 7), f"Failed for pred_len={pred_len}"

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 96, 7))).all()

    def test_gradient_flows(self):
        m = _make_model(revin=False)
        out = m(torch.randn(2, 96, 7))
        out.sum().backward()
        assert m.linear.weight.grad is not None

    def test_seq_len_not_multiple_of_period(self):
        """Padding should handle non-divisible seq_len."""
        m = SparseTSF(seq_len=100, pred_len=24, enc_in=3, period=24)
        out = m(torch.randn(2, 100, 3))
        assert out.shape == (2, 24, 3)

    def test_single_channel(self):
        m = SparseTSF(seq_len=48, pred_len=12, enc_in=1, period=12)
        out = m(torch.randn(4, 48, 1))
        assert out.shape == (4, 12, 1)

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 96, 7)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_period_1_output_shape(self):
        m = SparseTSF(seq_len=24, pred_len=12, enc_in=3, period=1, revin=False)
        out = m(torch.randn(2, 24, 3))
        assert out.shape == (2, 12, 3)


# ── registry ──────────────────────────────────────────────────────────────────


class TestSparseTSFRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import SparseTSF as S
        assert S is SparseTSF

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.SparseTSF import SparseTSFForecast
        assert SparseTSFForecast.model_type == "SparseTSF"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("SparseTSF", task)
            assert cls is not None
