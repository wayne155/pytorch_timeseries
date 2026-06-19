"""Tests for ETSformer — Exponential Smoothing Transformer."""
import torch
import torch.nn as nn
import math
import pytest

from torch_timeseries.model.ETSformer import (
    ETSformer,
    _ExponentialSmoothingAttention,
    _FrequencyAttention,
    _ETSformerLayer,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_model(seq_len=48, pred_len=12, enc_in=4, d_model=32, n_heads=2,
                e_layers=2, d_ff=64, dropout=0.0, top_k=3, revin=True):
    return ETSformer(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_model=d_model, n_heads=n_heads, e_layers=e_layers,
        d_ff=d_ff, dropout=dropout, top_k=top_k, revin=revin,
    )


# ── ExponentialSmoothingAttention ─────────────────────────────────────────────


class TestESAttn:
    def test_output_shape(self):
        attn = _ExponentialSmoothingAttention(d_model=32, n_heads=4)
        x = torch.randn(2, 24, 32)
        out = attn(x)
        assert out.shape == (2, 24, 32)

    def test_alpha_in_unit_interval(self):
        attn = _ExponentialSmoothingAttention(d_model=32, n_heads=4)
        alpha = torch.sigmoid(attn.log_alpha)
        assert (alpha > 0).all() and (alpha < 1).all()

    def test_smoothing_weights_causal(self):
        attn = _ExponentialSmoothingAttention(d_model=32, n_heads=2)
        T = 8
        w = attn._smoothing_weights(T, device=torch.device("cpu"))
        assert w.shape == (2, T, T)
        # Causal: w[:, t, s] == 0 for s > t
        for h in range(2):
            for t in range(T):
                for s in range(t + 1, T):
                    assert w[h, t, s].item() == 0.0

    def test_smoothing_weights_sum_to_one(self):
        attn = _ExponentialSmoothingAttention(d_model=32, n_heads=2)
        w = attn._smoothing_weights(8, device=torch.device("cpu"))
        # Each row should sum to 1 (after softmax)
        row_sums = w.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_gradient_flows(self):
        attn = _ExponentialSmoothingAttention(d_model=32, n_heads=2)
        x = torch.randn(2, 24, 32, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert attn.log_alpha.grad is not None


# ── FrequencyAttention ────────────────────────────────────────────────────────


class TestFAttn:
    def test_output_shape(self):
        attn = _FrequencyAttention(d_model=32, top_k=5)
        x = torch.randn(2, 48, 32)
        out = attn(x)
        assert out.shape == (2, 48, 32)

    def test_output_finite(self):
        attn = _FrequencyAttention(d_model=32, top_k=5)
        x = torch.randn(2, 48, 32)
        assert torch.isfinite(attn(x)).all()

    def test_gradient_flows(self):
        attn = _FrequencyAttention(d_model=32, top_k=5)
        x = torch.randn(2, 48, 32, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None


# ── ETSformerLayer ────────────────────────────────────────────────────────────


class TestETSformerLayer:
    def test_output_shape(self):
        layer = _ETSformerLayer(d_model=32, n_heads=2, d_ff=64, top_k=3)
        x = torch.randn(2, 48, 32)
        out = layer(x)
        assert out.shape == (2, 48, 32)

    def test_output_finite(self):
        layer = _ETSformerLayer(d_model=32, n_heads=2, d_ff=64, top_k=3)
        x = torch.randn(2, 48, 32)
        assert torch.isfinite(layer(x)).all()


# ── ETSformer full model ──────────────────────────────────────────────────────


class TestETSformerConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_num_layers(self):
        m = _make_model(e_layers=3)
        assert len(m.layers) == 3

    def test_revin_exists_when_enabled(self):
        m = _make_model(revin=True)
        assert hasattr(m, "revin_layer")

    def test_no_revin_when_disabled(self):
        m = _make_model(revin=False)
        assert not hasattr(m, "revin_layer")

    def test_input_proj_shape(self):
        m = _make_model(enc_in=4, d_model=32)
        assert m.input_proj.in_features == 4
        assert m.input_proj.out_features == 32

    def test_output_proj_shape(self):
        m = _make_model(pred_len=12, enc_in=4, d_model=32)
        assert m.output_proj.in_features == 32
        assert m.output_proj.out_features == 12 * 4


class TestETSformerForward:
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
            m = ETSformer(seq_len=96, pred_len=pred_len, enc_in=4, d_model=32, n_heads=2,
                          e_layers=1, d_ff=64, top_k=3)
            out = m(torch.randn(2, 96, 4))
            assert out.shape == (2, pred_len, 4), f"Failed for pred_len={pred_len}"

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 48, 4))).all()

    def test_single_channel(self):
        m = ETSformer(seq_len=24, pred_len=8, enc_in=1, d_model=16, n_heads=2,
                      e_layers=1, d_ff=32, top_k=3)
        out = m(torch.randn(4, 24, 1))
        assert out.shape == (4, 8, 1)

    def test_gradient_flows_to_alpha(self):
        m = _make_model(revin=False)
        out = m(torch.randn(2, 48, 4))
        out.sum().backward()
        # ES-Attn alpha should receive gradient
        assert m.layers[0].es_attn.log_alpha.grad is not None

    def test_gradient_flows_to_input(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 48, 4, requires_grad=True)
        out = m(x)
        out.sum().backward()
        assert x.grad is not None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 48, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_revin_tracks_scale(self):
        torch.manual_seed(0)
        m = _make_model(revin=True).eval()
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


# ── registry ──────────────────────────────────────────────────────────────────


class TestETSformerRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import ETSformer as E
        assert E is ETSformer

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.ETSformer import ETSformerForecast
        assert ETSformerForecast.model_type == "ETSformer"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("ETSformer", task)
            assert cls is not None
