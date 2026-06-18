"""Tests for torch_timeseries.nn FourierBlock and FourierCrossAttention."""
import pytest
import torch

from torch_timeseries.nn import FourierBlock, FourierCrossAttention

B, L, H, E = 4, 48, 8, 8   # batch, seq_len, n_heads, head_dim
D_MODEL = H * E             # 64


class TestFourierBlock:
    def test_output_shape(self):
        block = FourierBlock(
            in_channels=D_MODEL, out_channels=D_MODEL,
            seq_len=L, modes=16, mode_select_method='lowest', n_heads=H,
        )
        q = torch.randn(B, L, H, E)
        out, _ = block(q, q, q, mask=None)
        # output is (B, H, E, L) — permuted back in the caller (AttentionLayer)
        assert out.shape == (B, H, E, L)

    def test_no_nan(self):
        block = FourierBlock(
            in_channels=D_MODEL, out_channels=D_MODEL,
            seq_len=L, modes=8, mode_select_method='lowest', n_heads=H,
        )
        q = torch.randn(B, L, H, E)
        out, _ = block(q, q, q, mask=None)
        assert not torch.isnan(out).any()

    def test_gradients_flow(self):
        block = FourierBlock(
            in_channels=D_MODEL, out_channels=D_MODEL,
            seq_len=L, modes=8, mode_select_method='lowest', n_heads=H,
        )
        q = torch.randn(B, L, H, E, requires_grad=True)
        out, _ = block(q, q, q, mask=None)
        out.sum().backward()
        assert q.grad is not None

    def test_modes_clipped_to_seq_len_half(self):
        # modes > seq_len//2 should be silently clipped
        block = FourierBlock(
            in_channels=D_MODEL, out_channels=D_MODEL,
            seq_len=L, modes=1000, mode_select_method='lowest', n_heads=H,
        )
        q = torch.randn(B, L, H, E)
        out, _ = block(q, q, q, mask=None)
        assert out.shape == (B, H, E, L)


class TestFourierCrossAttention:
    def test_output_shape_same_seq(self):
        fca = FourierCrossAttention(
            in_channels=D_MODEL, out_channels=D_MODEL,
            seq_len_q=L, seq_len_kv=L,
            modes=8, mode_select_method='lowest', n_heads=H,
        )
        q = torch.randn(B, L, H, E)
        out, _ = fca(q, q, q, mask=None)
        assert out.shape == (B, H, E, L)

    def test_output_shape_different_seq(self):
        S = 24
        fca = FourierCrossAttention(
            in_channels=D_MODEL, out_channels=D_MODEL,
            seq_len_q=L, seq_len_kv=S,
            modes=8, mode_select_method='lowest', n_heads=H,
        )
        q = torch.randn(B, L, H, E)
        k = torch.randn(B, S, H, E)
        out, _ = fca(q, k, k, mask=None)
        assert out.shape == (B, H, E, L)

    def test_no_nan(self):
        fca = FourierCrossAttention(
            in_channels=D_MODEL, out_channels=D_MODEL,
            seq_len_q=L, seq_len_kv=L,
            modes=8, mode_select_method='lowest', n_heads=H,
        )
        q = torch.randn(B, L, H, E)
        out, _ = fca(q, q, q, mask=None)
        assert not torch.isnan(out).any()

    def test_softmax_activation(self):
        fca = FourierCrossAttention(
            in_channels=D_MODEL, out_channels=D_MODEL,
            seq_len_q=L, seq_len_kv=L,
            modes=8, mode_select_method='lowest',
            activation='softmax', n_heads=H,
        )
        q = torch.randn(B, L, H, E)
        out, _ = fca(q, q, q, mask=None)
        assert out.shape == (B, H, E, L)
        assert not torch.isnan(out).any()

    def test_gradients_flow(self):
        fca = FourierCrossAttention(
            in_channels=D_MODEL, out_channels=D_MODEL,
            seq_len_q=L, seq_len_kv=L,
            modes=8, mode_select_method='lowest', n_heads=H,
        )
        q = torch.randn(B, L, H, E, requires_grad=True)
        out, _ = fca(q, q, q, mask=None)
        out.sum().backward()
        assert q.grad is not None
