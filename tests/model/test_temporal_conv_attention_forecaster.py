"""Tests for TemporalConvAttentionForecaster — dilated causal TCN + temporal self-attention."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.TemporalConvAttentionForecaster import (
    TemporalConvAttentionForecaster,
    _CausalDilatedResBlock,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=32, pred_len=8, enc_in=4,
    d_model=32, n_heads=4, n_blocks=3, kernel_size=3, dropout=0.1, revin=True,
):
    return TemporalConvAttentionForecaster(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_model=d_model, n_heads=n_heads, n_blocks=n_blocks,
        kernel_size=kernel_size, dropout=dropout, revin=revin,
    )


# ── causal res block ────────────────────────────────────────────────────────────


class TestCausalDilatedResBlock:
    def test_output_shape_preserved(self):
        block = _CausalDilatedResBlock(d_model=16, kernel_size=3, dilation=1, dropout=0.0)
        x = torch.randn(4, 16, 32)
        assert block(x).shape == (4, 16, 32)

    def test_dilation_2_shape(self):
        block = _CausalDilatedResBlock(d_model=16, kernel_size=3, dilation=2, dropout=0.0)
        x = torch.randn(4, 16, 32)
        assert block(x).shape == (4, 16, 32)

    def test_residual_preserves_shape(self):
        block = _CausalDilatedResBlock(d_model=8, kernel_size=3, dilation=4, dropout=0.0)
        x = torch.randn(2, 8, 16)
        out = block(x)
        assert out.shape == x.shape


# ── construction ───────────────────────────────────────────────────────────────


class TestTemporalConvAttentionForecasterConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_num_tcn_blocks(self):
        m = _make_model(n_blocks=4)
        assert len(m.tcn_blocks) == 4

    def test_dilation_doubles(self):
        m = _make_model(n_blocks=4)
        dilations = [b.conv1.dilation[0] for b in m.tcn_blocks]
        assert dilations == [1, 2, 4, 8]

    def test_input_proj_shape(self):
        m = _make_model(d_model=32)
        assert m.input_proj.in_features == 1
        assert m.input_proj.out_features == 32

    def test_head_shape(self):
        m = _make_model(d_model=32, pred_len=12)
        assert m.head.in_features == 32
        assert m.head.out_features == 12


# ── forward ────────────────────────────────────────────────────────────────────


class TestTemporalConvAttentionForecasterForward:
    def test_output_shape(self):
        m = _make_model(seq_len=32, pred_len=8, enc_in=4)
        assert m(torch.randn(2, 32, 4)).shape == (2, 8, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        assert m(torch.randn(2, 32, 4)).shape == (2, 8, 4)

    def test_single_channel(self):
        m = TemporalConvAttentionForecaster(seq_len=16, pred_len=4, enc_in=1, d_model=16, n_heads=4)
        assert m(torch.randn(2, 16, 1)).shape == (2, 4, 1)

    def test_many_channels(self):
        m = TemporalConvAttentionForecaster(seq_len=16, pred_len=4, enc_in=8, d_model=16, n_heads=4)
        assert m(torch.randn(2, 16, 8)).shape == (2, 4, 8)

    def test_various_pred_lens(self):
        for pred_len in [8, 24, 96]:
            m = TemporalConvAttentionForecaster(seq_len=32, pred_len=pred_len, enc_in=4, d_model=16, n_heads=4)
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
        assert m.input_proj.weight.grad is not None
        assert m.tcn_blocks[0].conv1.weight.grad is not None
        assert m.head.weight.grad is not None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 32, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))


# ── registry ───────────────────────────────────────────────────────────────────


class TestTemporalConvAttentionForecasterRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import TemporalConvAttentionForecaster as M
        assert M is TemporalConvAttentionForecaster

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.TemporalConvAttentionForecaster import TemporalConvAttentionForecasterForecast
        assert TemporalConvAttentionForecasterForecast.model_type == "TemporalConvAttentionForecaster"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            assert get_experiment_class("TemporalConvAttentionForecaster", task) is not None
