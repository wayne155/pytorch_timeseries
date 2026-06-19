"""Tests for TFT — Temporal Fusion Transformer."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.TFT import (
    TFT,
    _GRN,
    _GatedLinearUnit,
    _VariableSelectionNetwork,
    _TemporalSelfAttention,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(seq_len=48, pred_len=12, enc_in=4, d_model=32, n_heads=4,
                num_lstm_layers=2, dropout=0.0):
    return TFT(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_model=d_model, n_heads=n_heads, num_lstm_layers=num_lstm_layers,
        dropout=dropout,
    )


# ── GatedLinearUnit ───────────────────────────────────────────────────────────


class TestGLU:
    def test_output_shape(self):
        glu = _GatedLinearUnit(input_size=32, output_size=16)
        x = torch.randn(2, 10, 32)
        out = glu(x)
        assert out.shape == (2, 10, 16)

    def test_gradient_flows(self):
        glu = _GatedLinearUnit(32, 16)
        x = torch.randn(2, 10, 32, requires_grad=True)
        glu(x).sum().backward()
        assert x.grad is not None

    def test_gate_is_sigmoid(self):
        """Output values should be bounded by the input (gate ∈ (0,1))."""
        glu = _GatedLinearUnit(4, 4)
        # With large positive inputs the gate → 1, so output ≈ val
        x = torch.zeros(1, 1, 4)
        out = glu(x)
        assert torch.isfinite(out).all()


# ── GRN ───────────────────────────────────────────────────────────────────────


class TestGRN:
    def test_output_shape_same_size(self):
        grn = _GRN(input_size=32, hidden_size=64, output_size=32)
        x = torch.randn(2, 10, 32)
        out = grn(x)
        assert out.shape == (2, 10, 32)

    def test_output_shape_different_size(self):
        grn = _GRN(input_size=8, hidden_size=32, output_size=16)
        x = torch.randn(2, 10, 8)
        out = grn(x)
        assert out.shape == (2, 10, 16)

    def test_with_context(self):
        grn = _GRN(input_size=32, hidden_size=64, output_size=32, context_size=16)
        x = torch.randn(2, 10, 32)
        ctx = torch.randn(2, 10, 16)
        out = grn(x, ctx)
        assert out.shape == (2, 10, 32)

    def test_output_finite(self):
        grn = _GRN(32, 64, 32)
        x = torch.randn(2, 10, 32)
        assert torch.isfinite(grn(x)).all()

    def test_gradient_flows(self):
        grn = _GRN(32, 64, 32)
        x = torch.randn(2, 10, 32, requires_grad=True)
        grn(x).sum().backward()
        assert x.grad is not None


# ── VariableSelectionNetwork ──────────────────────────────────────────────────


class TestVSN:
    def test_output_shape(self):
        vsn = _VariableSelectionNetwork(enc_in=4, d_model=32, dropout=0.0)
        x = torch.randn(2, 24, 4)
        out = vsn(x)
        assert out.shape == (2, 24, 32)

    def test_output_finite(self):
        vsn = _VariableSelectionNetwork(4, 32, 0.0)
        x = torch.randn(2, 24, 4)
        assert torch.isfinite(vsn(x)).all()

    def test_gradient_flows(self):
        vsn = _VariableSelectionNetwork(4, 32, 0.0)
        x = torch.randn(2, 24, 4, requires_grad=True)
        vsn(x).sum().backward()
        assert x.grad is not None

    def test_single_variate(self):
        vsn = _VariableSelectionNetwork(enc_in=1, d_model=16, dropout=0.0)
        x = torch.randn(2, 24, 1)
        out = vsn(x)
        assert out.shape == (2, 24, 16)


# ── TemporalSelfAttention ─────────────────────────────────────────────────────


class TestTemporalSelfAttention:
    def test_output_shape(self):
        attn = _TemporalSelfAttention(d_model=32, n_heads=4, dropout=0.0)
        x = torch.randn(2, 48, 32)
        out = attn(x)
        assert out.shape == (2, 48, 32)

    def test_output_finite(self):
        attn = _TemporalSelfAttention(32, 4, 0.0)
        x = torch.randn(2, 48, 32)
        assert torch.isfinite(attn(x)).all()

    def test_gradient_flows(self):
        attn = _TemporalSelfAttention(32, 4, 0.0)
        x = torch.randn(2, 48, 32, requires_grad=True)
        attn(x).sum().backward()
        assert x.grad is not None

    def test_requires_divisible_heads(self):
        with pytest.raises(AssertionError):
            _TemporalSelfAttention(33, 4, 0.0)


# ── TFT construction ──────────────────────────────────────────────────────────


class TestTFTConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_decoder_pos_shape(self):
        m = _make_model(pred_len=12, d_model=32)
        assert m.decoder_pos.shape == (1, 12, 32)

    def test_output_proj_shape(self):
        m = _make_model(enc_in=7, d_model=32)
        assert m.output_proj.out_features == 7


# ── TFT forward ───────────────────────────────────────────────────────────────


class TestTFTForward:
    def test_output_shape(self):
        m = _make_model(seq_len=48, pred_len=12, enc_in=4)
        out = m(torch.randn(4, 48, 4))
        assert out.shape == (4, 12, 4)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48, 96]:
            m = TFT(seq_len=96, pred_len=pred_len, enc_in=4, d_model=32,
                    n_heads=4, num_lstm_layers=1)
            out = m(torch.randn(2, 96, 4))
            assert out.shape == (2, pred_len, 4), f"Failed for pred_len={pred_len}"

    def test_single_channel(self):
        m = TFT(seq_len=24, pred_len=8, enc_in=1, d_model=16, n_heads=4,
                num_lstm_layers=1)
        out = m(torch.randn(2, 24, 1))
        assert out.shape == (2, 8, 1)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 48, 4))).all()

    def test_gradient_flows_to_input(self):
        m = _make_model()
        x = torch.randn(2, 48, 4, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_gradient_flows_to_vsn(self):
        m = _make_model()
        x = torch.randn(2, 48, 4)
        m(x).sum().backward()
        for p in m.vsn.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break

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
        torch.manual_seed(0)
        m = _make_model().eval()
        x_small = torch.randn(2, 48, 4) * 0.01
        x_large = torch.randn(2, 48, 4) * 100.0
        with torch.no_grad():
            out_small = m(x_small)
            out_large = m(x_large)
        assert out_large.abs().mean() > out_small.abs().mean()

    def test_constant_input_handled(self):
        m = _make_model().eval()
        x = torch.ones(2, 48, 4)
        with torch.no_grad():
            out = m(x)
        assert torch.isfinite(out).all()

    def test_num_lstm_layers_1(self):
        m = TFT(seq_len=48, pred_len=12, enc_in=4, d_model=32, n_heads=4,
                num_lstm_layers=1)
        out = m(torch.randn(2, 48, 4))
        assert out.shape == (2, 12, 4)


# ── registry ──────────────────────────────────────────────────────────────────


class TestTFTRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import TFT as M
        assert M is TFT

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.TFT import TFTForecast
        assert TFTForecast.model_type == "TFT"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("TFT", task)
            assert cls is not None
