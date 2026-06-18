"""Tests for torch_timeseries.nn.Patcher."""
import math
import pytest
import torch

from torch_timeseries.nn import Patcher

B, L, C = 4, 96, 7


# ── shape ─────────────────────────────────────────────────────────────────────

class TestPatcherShape:
    def test_non_overlapping_exact_fit(self):
        # L=96, p=16, s=16 → 96/16 = 6 patches
        p = Patcher(patch_len=16, stride=16, padding="none")
        x = torch.randn(B, L, C)
        out = p(x)
        assert out.shape == (B, 6, 16, C)

    def test_output_dims(self):
        p = Patcher(patch_len=16, stride=8)    # overlapping
        x = torch.randn(B, L, C)
        out = p(x)
        # (B, N, P, C)
        assert out.dim() == 4
        assert out.shape[0] == B
        assert out.shape[2] == 16
        assert out.shape[3] == C

    def test_num_patches_stored(self):
        p = Patcher(patch_len=16, stride=8)
        x = torch.randn(B, L, C)
        out = p(x)
        assert p.num_patches == out.shape[1]

    def test_stride_equals_patch_len_default(self):
        p = Patcher(patch_len=16)
        q = Patcher(patch_len=16, stride=16, padding="end")
        x = torch.randn(B, L, C)
        assert p(x).shape == q(x).shape

    def test_full_overlap_stride_1(self):
        p = Patcher(patch_len=4, stride=1, padding="none")
        x = torch.randn(B, 10, 3)
        out = p(x)
        expected_n = (10 - 4) // 1 + 1   # 7
        assert out.shape == (B, 7, 4, 3)

    def test_patch_len_equals_seq_len(self):
        # One patch covering the whole sequence
        p = Patcher(patch_len=L, stride=L, padding="none")
        x = torch.randn(B, L, C)
        out = p(x)
        assert out.shape == (B, 1, L, C)

    def test_single_channel(self):
        p = Patcher(patch_len=8, stride=4)
        x = torch.randn(B, 32, 1)
        out = p(x)
        assert out.shape[-1] == 1

    def test_batch_size_1(self):
        p = Patcher(patch_len=16, stride=8)
        x = torch.randn(1, L, C)
        out = p(x)
        assert out.shape[0] == 1


# ── padding modes ─────────────────────────────────────────────────────────────

class TestPatcherPadding:
    def test_end_padding_last_value_replicated(self):
        p = Patcher(patch_len=5, stride=5, padding="end")
        x = torch.arange(7, dtype=torch.float).reshape(1, 7, 1)
        # 7 / 5 = 1.4 → needs 10 steps → 3 extra steps repeating x[:,6,:]
        out = p(x)
        # Second patch: positions 5,6,pad,pad,pad = 5,6,6,6,6
        second_patch = out[0, 1, :, 0]
        assert second_patch[0].item() == 5.0
        assert second_patch[1].item() == 6.0
        assert (second_patch[2:] == 6.0).all()

    def test_constant_padding(self):
        p = Patcher(patch_len=5, stride=5, padding=0)
        x = torch.ones(1, 7, 1)
        out = p(x)
        # Second patch positions 5..9 → 5=1, 6=1, 7..9=0
        second = out[0, 1, :, 0]
        assert second[0].item() == 1.0
        assert second[1].item() == 1.0
        assert (second[2:] == 0.0).all()

    def test_none_padding_exact_fit(self):
        p = Patcher(patch_len=4, stride=4, padding="none")
        x = torch.randn(2, 12, 3)
        out = p(x)
        assert out.shape == (2, 3, 4, 3)

    def test_end_padding_increases_num_patches_vs_none(self):
        L = 10
        x = torch.randn(1, L, 2)
        p_end  = Patcher(patch_len=4, stride=4, padding="end")
        p_none = Patcher(patch_len=4, stride=4, padding="none")
        n_end  = p_end(x).shape[1]
        n_none = p_none(x).shape[1]
        assert n_end >= n_none


# ── value correctness ─────────────────────────────────────────────────────────

class TestPatcherValues:
    def test_patches_contain_correct_values(self):
        p = Patcher(patch_len=4, stride=4, padding="none")
        x = torch.arange(12, dtype=torch.float).reshape(1, 12, 1)
        out = p(x)   # (1, 3, 4, 1)
        expected_patch0 = torch.tensor([0., 1., 2., 3.])
        expected_patch1 = torch.tensor([4., 5., 6., 7.])
        expected_patch2 = torch.tensor([8., 9., 10., 11.])
        assert torch.allclose(out[0, 0, :, 0], expected_patch0)
        assert torch.allclose(out[0, 1, :, 0], expected_patch1)
        assert torch.allclose(out[0, 2, :, 0], expected_patch2)

    def test_overlapping_patches_share_values(self):
        p = Patcher(patch_len=4, stride=2, padding="none")
        x = torch.arange(10, dtype=torch.float).reshape(1, 10, 1)
        out = p(x)   # (1, 4, 4, 1)
        # Patch 0: [0,1,2,3], Patch 1: [2,3,4,5] → positions 2,3 shared
        assert torch.allclose(out[0, 0, 2:, 0], out[0, 1, :2, 0])

    def test_no_nan(self):
        p = Patcher(patch_len=16, stride=8)
        x = torch.randn(B, L, C)
        assert not torch.isnan(p(x)).any()

    def test_gradients_flow(self):
        p = Patcher(patch_len=16, stride=8)
        x = torch.randn(B, L, C, requires_grad=True)
        out = p(x)
        out.sum().backward()
        assert x.grad is not None


# ── edge cases ────────────────────────────────────────────────────────────────

class TestPatcherEdge:
    def test_invalid_patch_len_zero(self):
        with pytest.raises(AssertionError):
            Patcher(patch_len=0)

    def test_invalid_stride_zero(self):
        with pytest.raises(AssertionError):
            Patcher(patch_len=4, stride=0)

    def test_different_batch_sizes(self):
        p = Patcher(patch_len=16, stride=8)
        for b in [1, 2, 8, 16]:
            x = torch.randn(b, L, C)
            out = p(x)
            assert out.shape[0] == b
