"""Tests for NSTransformer — Non-stationary Transformer."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.NSTransformer import (
    NSTransformer,
    _DeStationaryAttention,
    _NSTransformerLayer,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(seq_len=48, pred_len=12, enc_in=4, d_model=32, n_heads=4,
                e_layers=2, d_ff=64, dropout=0.0):
    return NSTransformer(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_model=d_model, n_heads=n_heads, e_layers=e_layers,
        d_ff=d_ff, dropout=dropout,
    )


# ── DeStationaryAttention ──────────────────────────────────────────────────────


class TestDeStationaryAttention:
    def test_output_shape(self):
        attn = _DeStationaryAttention(d_model=32, n_heads=4)
        x = torch.randn(2, 24, 32)
        tau = torch.ones(2, 1, 1)
        delta = torch.zeros(2, 1, 1)
        out = attn(x, tau, delta)
        assert out.shape == (2, 24, 32)

    def test_output_finite(self):
        attn = _DeStationaryAttention(d_model=32, n_heads=4)
        x = torch.randn(2, 24, 32)
        tau = torch.ones(2, 1, 1)
        delta = torch.zeros(2, 1, 1)
        assert torch.isfinite(attn(x, tau, delta)).all()

    def test_gradient_flows(self):
        attn = _DeStationaryAttention(d_model=32, n_heads=4)
        x = torch.randn(2, 24, 32, requires_grad=True)
        tau = torch.ones(2, 1, 1, requires_grad=True)
        delta = torch.zeros(2, 1, 1, requires_grad=True)
        out = attn(x, tau, delta)
        out.sum().backward()
        assert x.grad is not None
        assert tau.grad is not None

    def test_tau_zero_different_from_tau_nonzero(self):
        torch.manual_seed(0)
        attn = _DeStationaryAttention(d_model=32, n_heads=4)
        x = torch.randn(2, 24, 32)
        out1 = attn(x, torch.zeros(2, 1, 1), torch.zeros(2, 1, 1))
        out2 = attn(x, torch.ones(2, 1, 1), torch.zeros(2, 1, 1))
        # With tau=0 all attention uniformly distributes; output should differ
        assert not torch.allclose(out1, out2, atol=1e-4)

    def test_requires_d_model_divisible_by_n_heads(self):
        with pytest.raises(AssertionError):
            _DeStationaryAttention(d_model=33, n_heads=4)


# ── NSTransformerLayer ─────────────────────────────────────────────────────────


class TestNSTransformerLayer:
    def test_output_shape(self):
        layer = _NSTransformerLayer(d_model=32, n_heads=4, d_ff=64, dropout=0.0)
        x = torch.randn(2, 24, 32)
        tau = torch.ones(2, 1, 1)
        delta = torch.zeros(2, 1, 1)
        out = layer(x, tau, delta)
        assert out.shape == (2, 24, 32)

    def test_output_finite(self):
        layer = _NSTransformerLayer(d_model=32, n_heads=4, d_ff=64, dropout=0.0)
        x = torch.randn(2, 48, 32)
        out = layer(x, torch.ones(2, 1, 1), torch.zeros(2, 1, 1))
        assert torch.isfinite(out).all()


# ── NSTransformer construction ─────────────────────────────────────────────────


class TestNSTransformerConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_num_layers(self):
        m = _make_model(e_layers=3)
        assert len(m.layers) == 3

    def test_input_proj_shape(self):
        m = _make_model(enc_in=4, d_model=32)
        assert m.input_proj.in_features == 4
        assert m.input_proj.out_features == 32

    def test_output_proj_shape(self):
        m = _make_model(pred_len=12, enc_in=4, d_model=32)
        assert m.output_proj.in_features == 32
        assert m.output_proj.out_features == 12 * 4

    def test_pos_embed_shape(self):
        m = _make_model(seq_len=48, d_model=32)
        assert m.pos_embed.shape == (1, 48, 32)


# ── NSTransformer forward ──────────────────────────────────────────────────────


class TestNSTransformerForward:
    def test_output_shape(self):
        m = _make_model(seq_len=48, pred_len=12, enc_in=4)
        out = m(torch.randn(4, 48, 4))
        assert out.shape == (4, 12, 4)

    def test_output_shape_single_channel(self):
        m = NSTransformer(seq_len=24, pred_len=8, enc_in=1, d_model=16, n_heads=4,
                          e_layers=1, d_ff=32)
        out = m(torch.randn(2, 24, 1))
        assert out.shape == (2, 8, 1)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48, 96]:
            m = NSTransformer(seq_len=96, pred_len=pred_len, enc_in=4, d_model=32,
                              n_heads=4, e_layers=1, d_ff=64)
            out = m(torch.randn(2, 96, 4))
            assert out.shape == (2, pred_len, 4), f"Failed for pred_len={pred_len}"

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 48, 4))).all()

    def test_gradient_flows(self):
        m = _make_model()
        x = torch.randn(2, 48, 4, requires_grad=True)
        out = m(x)
        out.sum().backward()
        assert x.grad is not None

    def test_gradient_flows_to_tau_proj(self):
        m = _make_model()
        x = torch.randn(2, 48, 4)
        out = m(x)
        out.sum().backward()
        for p in m.tau_proj.parameters():
            assert p.grad is not None

    def test_batch_size_1(self):
        m = _make_model()
        out = m(torch.randn(1, 48, 4))
        assert out.shape == (1, 12, 4)

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 48, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_denormalization_scales_output(self):
        """Output should scale with input magnitude (de-norm applied)."""
        torch.manual_seed(0)
        m = _make_model().eval()
        x_small = torch.randn(2, 48, 4) * 0.01
        x_large = torch.randn(2, 48, 4) * 100.0
        with torch.no_grad():
            out_small = m(x_small)
            out_large = m(x_large)
        assert out_large.abs().mean() > out_small.abs().mean()

    def test_constant_input_handled(self):
        """Constant series (std≈0) should not produce NaN due to clamping."""
        m = _make_model().eval()
        x = torch.ones(2, 48, 4)
        with torch.no_grad():
            out = m(x)
        assert torch.isfinite(out).all()


# ── registry ───────────────────────────────────────────────────────────────────


class TestNSTransformerRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import NSTransformer as M
        assert M is NSTransformer

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.NSTransformer import NSTransformerForecast
        assert NSTransformerForecast.model_type == "NSTransformer"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("NSTransformer", task)
            assert cls is not None
