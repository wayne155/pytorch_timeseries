"""Tests for BiLSTMForecaster — bidirectional LSTM with additive attention."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.BiLSTMForecaster import BiLSTMForecaster


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=32, pred_len=8, enc_in=4,
    d_model=32, num_layers=2, d_attn=16, dropout=0.1, revin=True,
):
    return BiLSTMForecaster(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_model=d_model, num_layers=num_layers,
        d_attn=d_attn, dropout=dropout, revin=revin,
    )


# ── construction ───────────────────────────────────────────────────────────────


class TestBiLSTMForecasterConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_lstm_is_bidirectional(self):
        m = _make_model(d_model=32)
        assert m.lstm.bidirectional

    def test_lstm_input_size_is_1(self):
        m = _make_model()
        assert m.lstm.input_size == 1

    def test_attention_weight_shape(self):
        m = _make_model(d_model=32, d_attn=16)
        # attn_w: (d_bi=64, d_attn=16)
        assert m.attn_w.weight.shape == (16, 64)

    def test_attention_v_shape(self):
        m = _make_model(d_attn=16)
        assert m.attn_v.weight.shape == (1, 16)

    def test_head_output_size(self):
        m = _make_model(d_model=32, pred_len=12)
        assert m.head.out_features == 12

    def test_head_input_size(self):
        m = _make_model(d_model=32)
        # d_bi = 2 * d_model = 64
        assert m.head.in_features == 64


# ── forward ────────────────────────────────────────────────────────────────────


class TestBiLSTMForecasterForward:
    def test_output_shape(self):
        m = _make_model(seq_len=32, pred_len=8, enc_in=4)
        assert m(torch.randn(2, 32, 4)).shape == (2, 8, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        assert m(torch.randn(2, 32, 4)).shape == (2, 8, 4)

    def test_single_channel(self):
        m = BiLSTMForecaster(seq_len=16, pred_len=4, enc_in=1)
        assert m(torch.randn(2, 16, 1)).shape == (2, 4, 1)

    def test_many_channels(self):
        m = BiLSTMForecaster(seq_len=16, pred_len=4, enc_in=8)
        assert m(torch.randn(2, 16, 8)).shape == (2, 4, 8)

    def test_various_pred_lens(self):
        for pred_len in [8, 24, 96]:
            m = BiLSTMForecaster(seq_len=32, pred_len=pred_len, enc_in=4, d_model=16)
            assert m(torch.randn(2, 32, 4)).shape == (2, pred_len, 4)

    def test_batch_size_1(self):
        m = _make_model()
        assert m(torch.randn(1, 32, 4)).shape == (1, 8, 4)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 32, 4))).all()

    def test_gradient_flows(self):
        m = _make_model()
        m(torch.randn(2, 32, 4)).sum().backward()
        assert m.lstm.weight_ih_l0.grad is not None
        assert m.attn_w.weight.grad is not None
        assert m.head.weight.grad is not None

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

    def test_attention_is_soft(self):
        """Attention weights should sum to 1 over time."""
        m = _make_model().eval()
        # Access attention weights directly via hook
        T = 32
        x_ci = torch.randn(1, T, 1)
        with torch.no_grad():
            out, _ = m.lstm(x_ci)
            energy = m.attn_v(torch.tanh(m.attn_w(out)))
            alpha = torch.softmax(energy, dim=1)
        assert abs(alpha.sum().item() - 1.0) < 1e-4


# ── registry ───────────────────────────────────────────────────────────────────


class TestBiLSTMForecasterRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import BiLSTMForecaster as M
        assert M is BiLSTMForecaster

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.BiLSTMForecaster import BiLSTMForecasterForecast
        assert BiLSTMForecasterForecast.model_type == "BiLSTMForecaster"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            assert get_experiment_class("BiLSTMForecaster", task) is not None
