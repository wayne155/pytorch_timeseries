"""Tests for MambaForecaster — selective state space model (Mamba/S6)."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.MambaForecaster import (
    MambaForecaster,
    _SelectiveSSM,
    _MambaBlock,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(seq_len=48, pred_len=12, enc_in=4, d_model=16, d_state=8,
                e_layers=2, dropout=0.0, revin=True):
    return MambaForecaster(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_model=d_model, d_state=d_state, e_layers=e_layers,
        dropout=dropout, revin=revin,
    )


# ── SelectiveSSM ──────────────────────────────────────────────────────────────


class TestSelectiveSSM:
    def test_output_shape(self):
        ssm = _SelectiveSSM(d_model=16, d_state=8)
        x = torch.randn(2, 24, 16)
        out = ssm(x)
        assert out.shape == (2, 24, 16)

    def test_output_finite(self):
        ssm = _SelectiveSSM(d_model=16, d_state=8)
        x = torch.randn(2, 24, 16)
        assert torch.isfinite(ssm(x)).all()

    def test_gradient_flows(self):
        ssm = _SelectiveSSM(d_model=16, d_state=8)
        x = torch.randn(2, 24, 16, requires_grad=True)
        ssm(x).sum().backward()
        assert x.grad is not None

    def test_A_log_gradient_flows(self):
        ssm = _SelectiveSSM(d_model=16, d_state=8)
        x = torch.randn(2, 24, 16)
        ssm(x).sum().backward()
        assert ssm.A_log.grad is not None

    def test_A_negative(self):
        """A must be negative for stability (log parameterised)."""
        ssm = _SelectiveSSM(d_model=16, d_state=8)
        A = -torch.exp(ssm.A_log)
        assert (A < 0).all()

    def test_different_seq_lens(self):
        ssm = _SelectiveSSM(d_model=16, d_state=4)
        for T in [8, 24, 48]:
            out = ssm(torch.randn(2, T, 16))
            assert out.shape == (2, T, 16)


# ── MambaBlock ────────────────────────────────────────────────────────────────


class TestMambaBlock:
    def test_output_shape(self):
        block = _MambaBlock(d_input=16, d_model=32, d_state=8, dropout=0.0)
        x = torch.randn(2, 24, 16)
        out = block(x)
        assert out.shape == (2, 24, 16)

    def test_output_finite(self):
        block = _MambaBlock(16, 32, 8, 0.0)
        x = torch.randn(2, 24, 16)
        assert torch.isfinite(block(x)).all()

    def test_gradient_flows(self):
        block = _MambaBlock(16, 32, 8, 0.0)
        x = torch.randn(2, 24, 16, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None


# ── MambaForecaster construction ───────────────────────────────────────────────


class TestMambaForecasterConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_num_blocks(self):
        m = _make_model(e_layers=3)
        assert len(m.blocks) == 3

    def test_revin_exists_when_enabled(self):
        m = _make_model(revin=True)
        assert hasattr(m, "revin_layer")

    def test_no_revin_when_disabled(self):
        m = _make_model(revin=False)
        assert not hasattr(m, "revin_layer")

    def test_output_proj_shape(self):
        m = _make_model(pred_len=12, enc_in=4, d_model=16)
        assert m.output_proj.out_features == 12 * 4


# ── MambaForecaster forward ────────────────────────────────────────────────────


class TestMambaForecasterForward:
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
            m = MambaForecaster(seq_len=96, pred_len=pred_len, enc_in=4,
                                d_model=16, d_state=8, e_layers=1)
            out = m(torch.randn(2, 96, 4))
            assert out.shape == (2, pred_len, 4), f"Failed pred_len={pred_len}"

    def test_single_channel(self):
        m = MambaForecaster(seq_len=24, pred_len=8, enc_in=1, d_model=8, d_state=4,
                            e_layers=1)
        out = m(torch.randn(2, 24, 1))
        assert out.shape == (2, 8, 1)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 48, 4))).all()

    def test_gradient_flows(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 48, 4, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_ssm_A_log_gradient_flows(self):
        m = _make_model()
        x = torch.randn(2, 48, 4)
        m(x).sum().backward()
        assert m.blocks[0].ssm.A_log.grad is not None

    def test_batch_size_1(self):
        m = _make_model()
        out = m(torch.randn(1, 48, 4))
        assert out.shape == (1, 12, 4)

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 48, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_revin_scales_output(self):
        torch.manual_seed(0)
        m = _make_model(revin=True).eval()
        x_small = torch.randn(2, 48, 4) * 0.01
        x_large = torch.randn(2, 48, 4) * 100.0
        with torch.no_grad():
            out_small = m(x_small)
            out_large = m(x_large)
        assert out_large.abs().mean() > out_small.abs().mean()

    def test_large_d_state(self):
        m = MambaForecaster(seq_len=48, pred_len=12, enc_in=4, d_model=32,
                            d_state=64, e_layers=1)
        out = m(torch.randn(2, 48, 4))
        assert out.shape == (2, 12, 4)


# ── registry ───────────────────────────────────────────────────────────────────


class TestMambaForecasterRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import MambaForecaster as M
        assert M is MambaForecaster

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.MambaForecaster import MambaForecasterForecast
        assert MambaForecasterForecast.model_type == "MambaForecaster"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("MambaForecaster", task)
            assert cls is not None
