"""Tests for torch_timeseries.nn.FeedForward and MixerBlock."""
import pytest
import torch

from torch_timeseries.nn import FeedForward, MixerBlock

B, T, C = 4, 96, 64


# ── FeedForward ───────────────────────────────────────────────────────────────

class TestFeedForward:
    def test_output_shape(self):
        ffn = FeedForward(d_model=C)
        x = torch.randn(B, T, C)
        assert ffn(x).shape == (B, T, C)

    def test_custom_d_ff(self):
        ffn = FeedForward(d_model=C, d_ff=256)
        x = torch.randn(B, T, C)
        assert ffn(x).shape == (B, T, C)

    def test_default_d_ff_is_4x(self):
        ffn = FeedForward(d_model=32)
        # inner layer has 4*32=128 neurons
        assert ffn.linear1.out_features == 128
        assert ffn.linear2.in_features == 128

    @pytest.mark.parametrize("act", ["relu", "gelu", "silu", "tanh"])
    def test_activations(self, act):
        ffn = FeedForward(d_model=C, activation=act)
        x = torch.randn(B, T, C)
        out = ffn(x)
        assert out.shape == (B, T, C)
        assert not torch.isnan(out).any()

    def test_module_activation(self):
        ffn = FeedForward(d_model=C, activation=torch.nn.ReLU())
        x = torch.randn(B, T, C)
        assert ffn(x).shape == (B, T, C)

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError):
            FeedForward(d_model=C, activation="swiglu")

    def test_gradients_flow(self):
        ffn = FeedForward(d_model=C)
        x = torch.randn(B, T, C, requires_grad=True)
        ffn(x).sum().backward()
        assert x.grad is not None
        assert ffn.linear1.weight.grad is not None

    def test_no_nan(self):
        ffn = FeedForward(d_model=C, dropout=0.0)
        x = torch.randn(B, T, C)
        assert not torch.isnan(ffn(x)).any()

    def test_arbitrary_batch_dims(self):
        # Works for (B, d_model) or (B, T, d_model)
        ffn = FeedForward(d_model=C)
        x2 = torch.randn(B, C)
        assert ffn(x2).shape == (B, C)

    def test_zero_dropout(self):
        ffn = FeedForward(d_model=C, dropout=0.0)
        ffn.eval()
        x = torch.randn(B, T, C)
        out1 = ffn(x)
        out2 = ffn(x)
        assert torch.allclose(out1, out2)


# ── MixerBlock ────────────────────────────────────────────────────────────────

class TestMixerBlock:
    def test_output_shape(self):
        block = MixerBlock(seq_len=T, d_model=C)
        x = torch.randn(B, T, C)
        assert block(x).shape == (B, T, C)

    def test_custom_hidden_dims(self):
        block = MixerBlock(seq_len=T, d_model=C, d_ff_time=128, d_ff_channel=128)
        x = torch.randn(B, T, C)
        assert block(x).shape == (B, T, C)

    def test_gradients_flow(self):
        block = MixerBlock(seq_len=T, d_model=C)
        x = torch.randn(B, T, C, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None

    def test_no_nan(self):
        block = MixerBlock(seq_len=T, d_model=C, dropout=0.0)
        x = torch.randn(B, T, C)
        assert not torch.isnan(block(x)).any()

    def test_residual_connection_when_zeroed(self):
        block = MixerBlock(seq_len=T, d_model=C)
        # Init all weights and biases to zero → output = input (residual passthrough)
        for p in block.parameters():
            p.data.zero_()
        block.eval()
        x = torch.randn(B, T, C)
        out = block(x)
        assert torch.allclose(out, x)

    def test_layernorm_applied(self):
        # verify norms exist
        block = MixerBlock(seq_len=T, d_model=C)
        assert isinstance(block.norm1, torch.nn.LayerNorm)
        assert isinstance(block.norm2, torch.nn.LayerNorm)

    def test_short_seq(self):
        block = MixerBlock(seq_len=8, d_model=16)
        x = torch.randn(2, 8, 16)
        assert block(x).shape == (2, 8, 16)

    def test_different_channel_counts(self):
        for c in [16, 32, 64, 128]:
            block = MixerBlock(seq_len=T, d_model=c)
            x = torch.randn(B, T, c)
            assert block(x).shape == (B, T, c)
