"""Tests for iMamba — inverted Mamba (SSM over the variate axis)."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.iMamba import (
    iMamba,
    _SelectiveSSM,
    _MambaBlock,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=96, pred_len=12, enc_in=4,
    d_model=32, d_state=8, e_layers=2, d_ff=64,
    dropout=0.0, revin=True,
):
    return iMamba(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_model=d_model, d_state=d_state, e_layers=e_layers,
        d_ff=d_ff, dropout=dropout, revin=revin,
    )


# ── SelectiveSSM ───────────────────────────────────────────────────────────────


class TestSelectiveSSM:
    def test_output_shape(self):
        ssm = _SelectiveSSM(16, 4)
        x = torch.randn(2, 8, 16)
        assert ssm(x).shape == (2, 8, 16)

    def test_A_negative(self):
        ssm = _SelectiveSSM(16, 4)
        assert (-torch.exp(ssm.A_log) < 0).all()

    def test_gradient_flows(self):
        ssm = _SelectiveSSM(16, 4)
        x = torch.randn(2, 5, 16, requires_grad=True)
        ssm(x).sum().backward()
        assert x.grad is not None


# ── MambaBlock ─────────────────────────────────────────────────────────────────


class TestMambaBlock:
    def test_output_shape(self):
        block = _MambaBlock(32, 8)
        x = torch.randn(2, 10, 32)
        assert block(x).shape == (2, 10, 32)

    def test_residual_path(self):
        block = _MambaBlock(16, 4)
        x = torch.randn(1, 4, 16)
        out = block(x)
        assert out.shape == x.shape


# ── iMamba construction ────────────────────────────────────────────────────────


class TestiMambaConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_embed_shape(self):
        m = _make_model(seq_len=96, d_model=32)
        assert m.embed.in_features == 96
        assert m.embed.out_features == 32

    def test_head_shape(self):
        m = _make_model(pred_len=24, d_model=32)
        assert m.head.in_features == 32
        assert m.head.out_features == 24

    def test_layer_count(self):
        m = _make_model(e_layers=4)
        assert len(m.mamba_layers) == 4


# ── iMamba forward ─────────────────────────────────────────────────────────────


class TestiMambaForward:
    def test_output_shape(self):
        m = _make_model(seq_len=96, pred_len=12, enc_in=4)
        out = m(torch.randn(2, 96, 4))
        assert out.shape == (2, 12, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        out = m(torch.randn(2, 96, 4))
        assert out.shape == (2, 12, 4)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48, 96]:
            m = iMamba(seq_len=96, pred_len=pred_len, enc_in=4,
                       d_model=32, d_state=8, e_layers=1, d_ff=64)
            out = m(torch.randn(2, 96, 4))
            assert out.shape == (2, pred_len, 4)

    def test_single_channel(self):
        m = iMamba(seq_len=48, pred_len=8, enc_in=1,
                   d_model=16, d_state=4, e_layers=1, d_ff=32)
        out = m(torch.randn(2, 48, 1))
        assert out.shape == (2, 8, 1)

    def test_many_channels(self):
        m = iMamba(seq_len=48, pred_len=12, enc_in=32,
                   d_model=32, d_state=8, e_layers=1, d_ff=64)
        out = m(torch.randn(2, 48, 32))
        assert out.shape == (2, 12, 32)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 96, 4))).all()

    def test_batch_size_1(self):
        m = _make_model()
        assert m(torch.randn(1, 96, 4)).shape == (1, 12, 4)

    def test_gradient_flows_to_input(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 96, 4, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_gradient_flows_to_ssm(self):
        m = _make_model()
        x = torch.randn(2, 96, 4)
        m(x).sum().backward()
        assert m.mamba_layers[0].ssm.A_log.grad is not None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 96, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_revin_scales_output(self):
        torch.manual_seed(0)
        m = _make_model(revin=True).eval()
        with torch.no_grad():
            out_s = m(torch.randn(2, 96, 4) * 0.01)
            out_l = m(torch.randn(2, 96, 4) * 100.0)
        assert out_l.abs().mean() > out_s.abs().mean()

    def test_processes_variate_axis(self):
        # With 1 channel, the SSM sees a sequence of length 1;
        # with 4 channels, length 4. Both should work.
        m1 = iMamba(seq_len=48, pred_len=8, enc_in=1,
                    d_model=16, d_state=4, e_layers=1, d_ff=32)
        m4 = iMamba(seq_len=48, pred_len=8, enc_in=4,
                    d_model=16, d_state=4, e_layers=1, d_ff=32)
        assert m1(torch.randn(2, 48, 1)).shape == (2, 8, 1)
        assert m4(torch.randn(2, 48, 4)).shape == (2, 8, 4)


# ── registry ───────────────────────────────────────────────────────────────────


class TestiMambaRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import iMamba as M
        assert M is iMamba

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.iMamba import iMambaForecast
        assert iMambaForecast.model_type == "iMamba"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("iMamba", task)
            assert cls is not None
