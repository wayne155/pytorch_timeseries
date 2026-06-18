"""Tests for torch_timeseries.nn TCN building blocks."""
import pytest
import torch

from torch_timeseries.nn import CausalConv1d, TemporalBlock, TemporalConvNet

B, C, L = 4, 16, 64


# ── CausalConv1d ──────────────────────────────────────────────────────────────

class TestCausalConv1d:
    def test_output_shape(self):
        conv = CausalConv1d(C, C, kernel_size=3)
        x = torch.randn(B, C, L)
        assert conv(x).shape == (B, C, L)

    def test_output_shape_with_dilation(self):
        conv = CausalConv1d(C, C, kernel_size=3, dilation=4)
        x = torch.randn(B, C, L)
        assert conv(x).shape == (B, C, L)

    def test_channel_change(self):
        conv = CausalConv1d(C, 32, kernel_size=3)
        x = torch.randn(B, C, L)
        assert conv(x).shape == (B, 32, L)

    def test_causality(self):
        # Changing input at position t should not affect output at position < t
        conv = CausalConv1d(1, 1, kernel_size=3, dilation=1)
        conv.eval()
        x = torch.zeros(1, 1, L)
        out_base = conv(x).detach()
        x2 = x.clone()
        x2[0, 0, L // 2] = 100.0   # spike at middle
        out_spike = conv(x2).detach()
        # Positions before the spike should be identical
        assert torch.allclose(out_base[0, 0, :L // 2], out_spike[0, 0, :L // 2])

    def test_gradients_flow(self):
        conv = CausalConv1d(C, C, kernel_size=3)
        x = torch.randn(B, C, L, requires_grad=True)
        out = conv(x)
        out.sum().backward()
        assert x.grad is not None
        assert conv.conv.weight.grad is not None

    def test_no_nan(self):
        conv = CausalConv1d(C, C, kernel_size=5, dilation=2)
        x = torch.randn(B, C, L)
        out = conv(x)
        assert not torch.isnan(out).any()

    def test_kernel_size_1(self):
        conv = CausalConv1d(C, C, kernel_size=1)
        x = torch.randn(B, C, L)
        assert conv(x).shape == (B, C, L)


# ── TemporalBlock ─────────────────────────────────────────────────────────────

class TestTemporalBlock:
    def test_output_shape_same_channels(self):
        block = TemporalBlock(C, C, kernel_size=3, dilation=1)
        x = torch.randn(B, C, L)
        assert block(x).shape == (B, C, L)

    def test_output_shape_channel_change(self):
        block = TemporalBlock(C, 32, kernel_size=3, dilation=2)
        x = torch.randn(B, C, L)
        assert block(x).shape == (B, 32, L)

    def test_residual_projection_used_when_channels_differ(self):
        block = TemporalBlock(C, 32, kernel_size=3, dilation=1)
        assert block.downsample is not None

    def test_no_residual_projection_same_channels(self):
        block = TemporalBlock(C, C, kernel_size=3, dilation=1)
        assert block.downsample is None

    def test_gradients_flow(self):
        block = TemporalBlock(C, C, kernel_size=3, dilation=1)
        x = torch.randn(B, C, L, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None

    def test_output_nonneg_relu(self):
        block = TemporalBlock(C, C, kernel_size=3, dilation=1, dropout=0.0)
        block.eval()
        x = torch.randn(B, C, L)
        out = block(x)
        assert (out >= 0).all()

    def test_no_nan(self):
        block = TemporalBlock(C, C, kernel_size=3, dilation=1)
        x = torch.randn(B, C, L)
        assert not torch.isnan(block(x)).any()


# ── TemporalConvNet ───────────────────────────────────────────────────────────

class TestTemporalConvNet:
    def test_output_shape(self):
        tcn = TemporalConvNet(C, [32, 32, 32], kernel_size=3)
        x = torch.randn(B, C, L)
        assert tcn(x).shape == (B, 32, L)

    def test_single_level(self):
        tcn = TemporalConvNet(C, [64], kernel_size=3)
        x = torch.randn(B, C, L)
        assert tcn(x).shape == (B, 64, L)

    def test_growing_channels(self):
        tcn = TemporalConvNet(7, [16, 32, 64], kernel_size=3)
        x = torch.randn(B, 7, L)
        assert tcn(x).shape == (B, 64, L)

    def test_depth_4_works(self):
        tcn = TemporalConvNet(C, [C] * 4, kernel_size=2)
        x = torch.randn(B, C, L)
        assert tcn(x).shape == (B, C, L)

    def test_receptive_field_exponential(self):
        # With kernel=2, 2 convs/block, 4 levels (d=1,2,4,8):
        # RF = 1 + 2*(1+2+4+8) = 31
        # A spike at position 0 should NOT affect positions >= 31.
        tcn = TemporalConvNet(1, [1, 1, 1, 1], kernel_size=2, dropout=0.0)
        tcn.eval()
        with torch.no_grad():
            x_spike = torch.zeros(1, 1, 64)
            x_spike[0, 0, 0] = 1.0
            out_spike = tcn(x_spike)
            out_zero  = tcn(torch.zeros(1, 1, 64))
            assert torch.allclose(out_spike[0, 0, 31:], out_zero[0, 0, 31:], atol=1e-5)

    def test_gradients_flow(self):
        tcn = TemporalConvNet(C, [32, 32], kernel_size=3)
        x = torch.randn(B, C, L, requires_grad=True)
        out = tcn(x)
        out.sum().backward()
        assert x.grad is not None

    def test_no_nan(self):
        tcn = TemporalConvNet(C, [32, 32, 32], kernel_size=3)
        x = torch.randn(B, C, L)
        assert not torch.isnan(tcn(x)).any()

    def test_channels_last_via_transpose(self):
        tcn = TemporalConvNet(7, [32, 32], kernel_size=3, dropout=0.0)
        tcn.eval()
        x_blc = torch.randn(B, L, 7)                     # (B, L, C)
        out_blc = tcn(x_blc.transpose(1, 2)).transpose(1, 2)  # back to (B, L, 32)
        assert out_blc.shape == (B, L, 32)
