"""Tests for torch_timeseries.nn.Conv_Blocks — Inception_Block_V1 and V2."""
import pytest
import torch

from torch_timeseries.nn.Conv_Blocks import Inception_Block_V1, Inception_Block_V2

B, C_IN, C_OUT, H, W = 4, 16, 32, 24, 48   # 2-D feature map


class TestInceptionBlockV1:
    def test_output_shape(self):
        block = Inception_Block_V1(in_channels=C_IN, out_channels=C_OUT, num_kernels=6)
        x = torch.randn(B, C_IN, H, W)
        out = block(x)
        assert out.shape == (B, C_OUT, H, W)

    def test_output_shape_same_channels(self):
        block = Inception_Block_V1(in_channels=C_IN, out_channels=C_IN, num_kernels=4)
        x = torch.randn(B, C_IN, H, W)
        assert block(x).shape == (B, C_IN, H, W)

    def test_num_kernels_1(self):
        block = Inception_Block_V1(in_channels=C_IN, out_channels=C_OUT, num_kernels=1)
        x = torch.randn(B, C_IN, H, W)
        assert block(x).shape == (B, C_OUT, H, W)

    def test_no_nan(self):
        block = Inception_Block_V1(in_channels=C_IN, out_channels=C_OUT)
        assert not torch.isnan(block(torch.randn(B, C_IN, H, W))).any()

    def test_gradients_flow(self):
        block = Inception_Block_V1(in_channels=C_IN, out_channels=C_OUT)
        x = torch.randn(B, C_IN, H, W, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None

    def test_init_weight_false(self):
        block = Inception_Block_V1(in_channels=C_IN, out_channels=C_OUT,
                                   num_kernels=3, init_weight=False)
        assert block(torch.randn(B, C_IN, H, W)).shape == (B, C_OUT, H, W)


class TestInceptionBlockV2:
    def test_output_shape(self):
        block = Inception_Block_V2(in_channels=C_IN, out_channels=C_OUT, num_kernels=6)
        x = torch.randn(B, C_IN, H, W)
        out = block(x)
        assert out.shape == (B, C_OUT, H, W)

    def test_output_shape_same_channels(self):
        block = Inception_Block_V2(in_channels=C_IN, out_channels=C_IN, num_kernels=4)
        assert block(torch.randn(B, C_IN, H, W)).shape == (B, C_IN, H, W)

    def test_no_nan(self):
        block = Inception_Block_V2(in_channels=C_IN, out_channels=C_OUT)
        assert not torch.isnan(block(torch.randn(B, C_IN, H, W))).any()

    def test_gradients_flow(self):
        block = Inception_Block_V2(in_channels=C_IN, out_channels=C_OUT)
        x = torch.randn(B, C_IN, H, W, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None

    def test_init_weight_false(self):
        block = Inception_Block_V2(in_channels=C_IN, out_channels=C_OUT,
                                   num_kernels=4, init_weight=False)
        assert block(torch.randn(B, C_IN, H, W)).shape == (B, C_OUT, H, W)
