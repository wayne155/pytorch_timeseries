"""Tests for WaveNet — dilated causal convolution forecasting model."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.WaveNet import WaveNet, _WaveNetBlock


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_model(seq_len=96, pred_len=24, enc_in=7, d_model=32, d_skip=32,
                kernel_size=2, num_layers=4, num_stacks=1, revin=True):
    return WaveNet(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_model=d_model, d_skip=d_skip, kernel_size=kernel_size,
        num_layers=num_layers, num_stacks=num_stacks, revin=revin,
    )


# ── WaveNetBlock ──────────────────────────────────────────────────────────────


class TestWaveNetBlock:
    def test_output_shapes(self):
        B, D, T = 4, 32, 96
        block = _WaveNetBlock(d_model=D, d_skip=D, kernel_size=2, dilation=1)
        x = torch.randn(B, D, T)
        residual, skip = block(x)
        assert residual.shape == (B, D, T)
        assert skip.shape == (B, D, T)

    def test_causal_padding_preserves_length(self):
        """Dilated causal conv must not change the time dimension."""
        for dilation in (1, 2, 4, 8):
            block = _WaveNetBlock(d_model=16, d_skip=16, kernel_size=2, dilation=dilation)
            x = torch.randn(2, 16, 96)
            res, skip = block(x)
            assert res.shape[2] == 96, f"dilation={dilation}: T changed"
            assert skip.shape[2] == 96

    def test_gradient_flows(self):
        block = _WaveNetBlock(d_model=16, d_skip=16, kernel_size=2, dilation=1)
        x = torch.randn(2, 16, 96, requires_grad=True)
        res, skip = block(x)
        (res + skip).sum().backward()
        assert x.grad is not None

    def test_output_finite(self):
        block = _WaveNetBlock(d_model=16, d_skip=16, kernel_size=2, dilation=4)
        x = torch.randn(2, 16, 96)
        res, skip = block(x)
        assert torch.isfinite(res).all()
        assert torch.isfinite(skip).all()


# ── construction ──────────────────────────────────────────────────────────────


class TestWaveNetConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_block_count(self):
        # 1 stack × 4 layers = 4 blocks
        m = _make_model(num_layers=4, num_stacks=1)
        assert len(m.blocks) == 4

    def test_multi_stack_block_count(self):
        m = _make_model(num_layers=4, num_stacks=3)
        assert len(m.blocks) == 12

    def test_revin_exists_when_enabled(self):
        m = _make_model(revin=True)
        assert hasattr(m, "revin_layer")

    def test_no_revin_when_disabled(self):
        m = _make_model(revin=False)
        assert not hasattr(m, "revin_layer")

    def test_exponential_dilations(self):
        m = WaveNet(seq_len=96, pred_len=24, enc_in=4, d_model=16, d_skip=16,
                    kernel_size=2, num_layers=4, num_stacks=1)
        expected_dilations = [1, 2, 4, 8]
        actual_dilations = [b.dilation for b in m.blocks]
        assert actual_dilations == expected_dilations


# ── forward ───────────────────────────────────────────────────────────────────


class TestWaveNetForward:
    def test_output_shape(self):
        m = _make_model(seq_len=96, pred_len=24, enc_in=7)
        out = m(torch.randn(4, 96, 7))
        assert out.shape == (4, 24, 7)

    def test_output_shape_no_revin(self):
        m = _make_model(seq_len=96, pred_len=24, enc_in=7, revin=False)
        out = m(torch.randn(4, 96, 7))
        assert out.shape == (4, 24, 7)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48, 96]:
            m = WaveNet(seq_len=96, pred_len=pred_len, enc_in=7, d_model=16, d_skip=16,
                        kernel_size=2, num_layers=4, num_stacks=1)
            out = m(torch.randn(2, 96, 7))
            assert out.shape == (2, pred_len, 7), f"Failed for pred_len={pred_len}"

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 96, 7))).all()

    def test_single_channel(self):
        m = WaveNet(seq_len=48, pred_len=12, enc_in=1, d_model=16, d_skip=16,
                    kernel_size=2, num_layers=4)
        out = m(torch.randn(4, 48, 1))
        assert out.shape == (4, 12, 1)

    def test_many_channels(self):
        m = WaveNet(seq_len=48, pred_len=12, enc_in=137, d_model=16, d_skip=16,
                    kernel_size=2, num_layers=4)
        out = m(torch.randn(2, 48, 137))
        assert out.shape == (2, 12, 137)

    def test_gradient_flows_to_input(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 96, 7, requires_grad=True)
        out = m(x)
        out.sum().backward()
        assert x.grad is not None

    def test_gradient_flows_to_blocks(self):
        m = _make_model(revin=False)
        out = m(torch.randn(2, 96, 7))
        out.sum().backward()
        assert m.blocks[0].conv_dil.weight.grad is not None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 96, 7)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_batch_size_1(self):
        m = _make_model()
        out = m(torch.randn(1, 96, 7))
        assert out.shape == (1, 24, 7)

    def test_two_stacks(self):
        m = _make_model(num_stacks=2)
        out = m(torch.randn(2, 96, 7))
        assert out.shape == (2, 24, 7)

    def test_revin_tracks_input_scale(self):
        torch.manual_seed(0)
        m = _make_model(revin=True).eval()
        x_small = torch.randn(2, 96, 7) * 0.01
        x_large = torch.randn(2, 96, 7) * 100.0
        with torch.no_grad():
            out_small = m(x_small)
            out_large = m(x_large)
        assert out_large.abs().mean() > out_small.abs().mean()


# ── registry ──────────────────────────────────────────────────────────────────


class TestWaveNetRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import WaveNet as W
        assert W is WaveNet

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.WaveNet import WaveNetForecast
        assert WaveNetForecast.model_type == "WaveNet"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("WaveNet", task)
            assert cls is not None
