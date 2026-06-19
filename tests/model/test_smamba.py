"""Tests for S-Mamba — intra-variate Mamba + inter-variate Transformer."""
import math

import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.SMamba import (
    SMamba,
    _SelectiveSSM,
    _MambaBlock,
    _InterVariateTransformer,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=64, pred_len=12, enc_in=4,
    d_model=32, d_state=8, e_layers=2, n_heads=4, d_ff=64,
    patch_len=8, stride=8, dropout=0.0, revin=True,
):
    return SMamba(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_model=d_model, d_state=d_state, e_layers=e_layers,
        n_heads=n_heads, d_ff=d_ff, patch_len=patch_len, stride=stride,
        dropout=dropout, revin=revin,
    )


# ── SelectiveSSM ───────────────────────────────────────────────────────────────


class TestSelectiveSSM:
    def test_output_shape(self):
        ssm = _SelectiveSSM(d_model=32, d_state=8)
        x = torch.randn(2, 10, 32)
        out = ssm(x)
        assert out.shape == (2, 10, 32)

    def test_gradient_flows(self):
        ssm = _SelectiveSSM(d_model=16, d_state=4)
        x = torch.randn(2, 5, 16, requires_grad=True)
        ssm(x).sum().backward()
        assert x.grad is not None

    def test_A_negative(self):
        ssm = _SelectiveSSM(d_model=16, d_state=4)
        A = -torch.exp(ssm.A_log)
        assert (A < 0).all()

    def test_skip_connection_present(self):
        ssm = _SelectiveSSM(d_model=16, d_state=4)
        assert ssm.D.shape == (16,)


# ── MambaBlock ─────────────────────────────────────────────────────────────────


class TestMambaBlock:
    def test_output_shape(self):
        block = _MambaBlock(d_model=32, d_state=8)
        x = torch.randn(2, 10, 32)
        assert block(x).shape == (2, 10, 32)

    def test_residual_identity_at_zero_params(self):
        block = _MambaBlock(d_model=16, d_state=4)
        x = torch.randn(2, 5, 16)
        out = block(x)
        assert out.shape == (2, 5, 16)


# ── InterVariateTransformer ────────────────────────────────────────────────────


class TestInterVariateTransformer:
    def test_output_shape(self):
        layer = _InterVariateTransformer(d_model=32, n_heads=4, d_ff=64)
        x = torch.randn(2, 6, 32)  # B, C, d_model
        assert layer(x).shape == (2, 6, 32)

    def test_gradient_flows(self):
        layer = _InterVariateTransformer(d_model=16, n_heads=2, d_ff=32)
        x = torch.randn(2, 4, 16, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None


# ── SMamba construction ────────────────────────────────────────────────────────


class TestSMambaConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_n_patches_correct(self):
        m = _make_model(seq_len=64, patch_len=8, stride=8)
        assert m.n_patches == math.ceil((64 - 8) / 8) + 1

    def test_layer_counts(self):
        m = _make_model(e_layers=3)
        assert len(m.intra_layers) == 3
        assert len(m.inter_layers) == 3

    def test_head_out_features(self):
        m = _make_model(pred_len=24)
        assert m.head.out_features == 24


# ── SMamba forward ─────────────────────────────────────────────────────────────


class TestSMambaForward:
    def test_output_shape(self):
        m = _make_model(seq_len=64, pred_len=12, enc_in=4)
        out = m(torch.randn(2, 64, 4))
        assert out.shape == (2, 12, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        out = m(torch.randn(2, 64, 4))
        assert out.shape == (2, 12, 4)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48]:
            m = SMamba(seq_len=96, pred_len=pred_len, enc_in=4,
                       d_model=16, d_state=4, e_layers=1, n_heads=4,
                       d_ff=32, patch_len=16, stride=16)
            out = m(torch.randn(2, 96, 4))
            assert out.shape == (2, pred_len, 4)

    def test_single_channel(self):
        m = SMamba(seq_len=32, pred_len=8, enc_in=1,
                   d_model=16, d_state=4, e_layers=1, n_heads=2,
                   d_ff=32, patch_len=8, stride=8)
        out = m(torch.randn(2, 32, 1))
        assert out.shape == (2, 8, 1)

    def test_many_channels(self):
        m = SMamba(seq_len=32, pred_len=8, enc_in=16,
                   d_model=16, d_state=4, e_layers=1, n_heads=4,
                   d_ff=32, patch_len=8, stride=8)
        out = m(torch.randn(2, 32, 16))
        assert out.shape == (2, 8, 16)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 64, 4))).all()

    def test_batch_size_1(self):
        m = _make_model()
        out = m(torch.randn(1, 64, 4))
        assert out.shape == (1, 12, 4)

    def test_gradient_flows_to_input(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 64, 4, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_gradient_flows_to_ssm(self):
        m = _make_model()
        x = torch.randn(2, 64, 4)
        m(x).sum().backward()
        assert m.intra_layers[0].ssm.A_log.grad is not None

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

    def test_single_layer(self):
        m = SMamba(seq_len=32, pred_len=8, enc_in=4,
                   d_model=16, d_state=4, e_layers=1, n_heads=2,
                   d_ff=32, patch_len=8, stride=8)
        out = m(torch.randn(2, 32, 4))
        assert out.shape == (2, 8, 4)


# ── registry ───────────────────────────────────────────────────────────────────


class TestSMambaRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import SMamba as M
        assert M is SMamba

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.SMamba import SMambaForecast
        assert SMambaForecast.model_type == "SMamba"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("SMamba", task)
            assert cls is not None
