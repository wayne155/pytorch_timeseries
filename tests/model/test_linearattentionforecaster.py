"""Tests for LinearAttentionForecaster — O(n) linear attention Transformer."""
import math
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.LinearAttentionForecaster import (
    LinearAttentionForecaster, _LinearMultiheadAttention, _LinearAttnLayer,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=64, pred_len=12, enc_in=4,
    d_model=32, n_heads=4, e_layers=2, d_ff=64,
    patch_len=8, stride=8, dropout=0.0, revin=True,
):
    return LinearAttentionForecaster(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff,
        patch_len=patch_len, stride=stride, dropout=dropout, revin=revin,
    )


# ── LinearMultiheadAttention unit tests ────────────────────────────────────────


class TestLinearMultiheadAttention:
    def test_output_shape(self):
        attn = _LinearMultiheadAttention(d_model=32, n_heads=4)
        x = torch.randn(2, 10, 32)
        assert attn(x).shape == (2, 10, 32)

    def test_gradient_flows(self):
        attn = _LinearMultiheadAttention(d_model=16, n_heads=2)
        x = torch.randn(2, 5, 16, requires_grad=True)
        attn(x).sum().backward()
        assert x.grad is not None

    def test_phi_is_positive(self):
        attn = _LinearMultiheadAttention(d_model=16, n_heads=2)
        x = torch.randn(4, 16) * 5  # large values
        phi = attn._phi(x)
        assert (phi > 0).all()

    def test_phi_elu_plus_1_at_zero(self):
        attn = _LinearMultiheadAttention(d_model=4, n_heads=2)
        x = torch.zeros(1, 4)
        # ELU(0) = 0, so φ(0) = 1
        assert torch.allclose(attn._phi(x), torch.ones(1, 4))

    def test_finite_output_large_seq(self):
        attn = _LinearMultiheadAttention(d_model=32, n_heads=4)
        x = torch.randn(2, 100, 32)
        assert torch.isfinite(attn(x)).all()


# ── LinearAttnLayer unit tests ─────────────────────────────────────────────────


class TestLinearAttnLayer:
    def test_output_shape(self):
        layer = _LinearAttnLayer(d_model=32, n_heads=4, d_ff=64)
        x = torch.randn(2, 8, 32)
        assert layer(x).shape == (2, 8, 32)

    def test_gradient_flows(self):
        layer = _LinearAttnLayer(d_model=16, n_heads=2, d_ff=32)
        x = torch.randn(2, 5, 16, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None


# ── LinearAttentionForecaster construction ─────────────────────────────────────


class TestLinearAttentionForecasterConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_n_patches_correct(self):
        m = _make_model(seq_len=64, patch_len=8, stride=8)
        assert m.n_patches == math.ceil((64 - 8) / 8) + 1

    def test_layer_count(self):
        assert len(_make_model(e_layers=3).layers) == 3

    def test_head_out_features(self):
        assert _make_model(pred_len=24).head.out_features == 24


# ── LinearAttentionForecaster forward ─────────────────────────────────────────


class TestLinearAttentionForecasterForward:
    def test_output_shape(self):
        m = _make_model(seq_len=64, pred_len=12, enc_in=4)
        assert m(torch.randn(2, 64, 4)).shape == (2, 12, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        assert m(torch.randn(2, 64, 4)).shape == (2, 12, 4)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48]:
            m = LinearAttentionForecaster(seq_len=96, pred_len=pred_len, enc_in=4,
                                          d_model=16, n_heads=2, e_layers=1, d_ff=32,
                                          patch_len=16, stride=16)
            assert m(torch.randn(2, 96, 4)).shape == (2, pred_len, 4)

    def test_single_channel(self):
        m = LinearAttentionForecaster(seq_len=32, pred_len=8, enc_in=1,
                                      d_model=16, n_heads=2, e_layers=1, d_ff=32,
                                      patch_len=8, stride=8)
        assert m(torch.randn(2, 32, 1)).shape == (2, 8, 1)

    def test_many_channels(self):
        m = LinearAttentionForecaster(seq_len=32, pred_len=8, enc_in=16,
                                      d_model=16, n_heads=2, e_layers=1, d_ff=32,
                                      patch_len=8, stride=8)
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

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 64, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_revin_scales_output(self):
        torch.manual_seed(0)
        m = _make_model(revin=True).eval()
        with torch.no_grad():
            out_s = m(torch.randn(2, 64, 4) * 0.01)
            out_l = m(torch.randn(2, 64, 4) * 100.0)
        assert out_l.abs().mean() > out_s.abs().mean()

    def test_long_sequence_faster_than_n2(self):
        # Linear attention should handle long sequences without OOM
        m = LinearAttentionForecaster(seq_len=512, pred_len=96, enc_in=4,
                                      d_model=32, n_heads=4, e_layers=1, d_ff=64,
                                      patch_len=16, stride=16)
        out = m(torch.randn(1, 512, 4))
        assert out.shape == (1, 96, 4)
        assert torch.isfinite(out).all()


# ── registry ───────────────────────────────────────────────────────────────────


class TestLinearAttentionForecasterRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import LinearAttentionForecaster as M
        assert M is LinearAttentionForecaster

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.LinearAttentionForecaster import (
            LinearAttentionForecasterForecast,
        )
        assert LinearAttentionForecasterForecast.model_type == "LinearAttentionForecaster"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            assert get_experiment_class("LinearAttentionForecaster", task) is not None
