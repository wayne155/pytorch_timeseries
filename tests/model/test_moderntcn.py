"""Tests for ModernTCN — large-kernel depthwise conv with patch embedding."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.ModernTCN import (
    ModernTCN,
    _ModernTCNBlock,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(seq_len=96, pred_len=12, enc_in=4, patch_size=8, patch_stride=4,
                d_model=32, kernel_size=11, e_layers=2, d_ff_ratio=4,
                dropout=0.0, revin=True):
    return ModernTCN(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        patch_size=patch_size, patch_stride=patch_stride,
        d_model=d_model, kernel_size=kernel_size, e_layers=e_layers,
        d_ff_ratio=d_ff_ratio, dropout=dropout, revin=revin,
    )


# ── ModernTCNBlock ─────────────────────────────────────────────────────────────


class TestModernTCNBlock:
    def test_output_shape_preserved(self):
        block = _ModernTCNBlock(d_model=32, kernel_size=11, d_ff=128, dropout=0.0)
        x = torch.randn(4, 32, 24)
        out = block(x)
        assert out.shape == (4, 32, 24), "block must preserve T_patch dimension"

    def test_output_finite(self):
        block = _ModernTCNBlock(32, 11, 128, 0.0)
        x = torch.randn(4, 32, 24)
        assert torch.isfinite(block(x)).all()

    def test_gradient_flows(self):
        block = _ModernTCNBlock(32, 11, 128, 0.0)
        x = torch.randn(4, 32, 24, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None

    def test_requires_odd_kernel(self):
        with pytest.raises(AssertionError):
            _ModernTCNBlock(32, 10, 128, 0.0)

    def test_large_kernel(self):
        block = _ModernTCNBlock(32, 51, 128, 0.0)
        x = torch.randn(2, 32, 48)
        out = block(x)
        assert out.shape == (2, 32, 48)


# ── ModernTCN construction ─────────────────────────────────────────────────────


class TestModernTCNConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_num_blocks(self):
        m = _make_model(e_layers=4)
        assert len(m.blocks) == 4

    def test_revin_exists_when_enabled(self):
        m = _make_model(revin=True)
        assert hasattr(m, "revin_layer")

    def test_no_revin_when_disabled(self):
        m = _make_model(revin=False)
        assert not hasattr(m, "revin_layer")

    def test_output_proj_shape(self):
        m = _make_model(pred_len=12, d_model=32)
        assert m.output_proj.out_features == 12


# ── ModernTCN forward ──────────────────────────────────────────────────────────


class TestModernTCNForward:
    def test_output_shape(self):
        m = _make_model(seq_len=96, pred_len=12, enc_in=4)
        out = m(torch.randn(4, 96, 4))
        assert out.shape == (4, 12, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        out = m(torch.randn(4, 96, 4))
        assert out.shape == (4, 12, 4)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48, 96]:
            m = ModernTCN(seq_len=96, pred_len=pred_len, enc_in=4,
                          patch_size=8, patch_stride=4, d_model=32,
                          kernel_size=11, e_layers=1, d_ff_ratio=2)
            out = m(torch.randn(2, 96, 4))
            assert out.shape == (2, pred_len, 4), f"Failed pred_len={pred_len}"

    def test_single_channel(self):
        m = ModernTCN(seq_len=48, pred_len=12, enc_in=1, patch_size=4,
                      patch_stride=2, d_model=16, kernel_size=7, e_layers=1,
                      d_ff_ratio=2)
        out = m(torch.randn(2, 48, 1))
        assert out.shape == (2, 12, 1)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 96, 4))).all()

    def test_gradient_flows(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 96, 4, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_batch_size_1(self):
        m = _make_model()
        out = m(torch.randn(1, 96, 4))
        assert out.shape == (1, 12, 4)

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 96, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_revin_scales_output(self):
        torch.manual_seed(0)
        m = _make_model(revin=True).eval()
        x_small = torch.randn(2, 96, 4) * 0.01
        x_large = torch.randn(2, 96, 4) * 100.0
        with torch.no_grad():
            out_small = m(x_small)
            out_large = m(x_large)
        assert out_large.abs().mean() > out_small.abs().mean()

    def test_patch_stride_1(self):
        m = ModernTCN(seq_len=48, pred_len=12, enc_in=4, patch_size=5,
                      patch_stride=1, d_model=16, kernel_size=7, e_layers=1,
                      d_ff_ratio=2)
        out = m(torch.randn(2, 48, 4))
        assert out.shape == (2, 12, 4)


# ── registry ───────────────────────────────────────────────────────────────────


class TestModernTCNRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import ModernTCN as M
        assert M is ModernTCN

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.ModernTCN import ModernTCNForecast
        assert ModernTCNForecast.model_type == "ModernTCN"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("ModernTCN", task)
            assert cls is not None
