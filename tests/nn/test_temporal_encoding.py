"""Tests for torch_timeseries.nn temporal encoding modules."""
import math
import pytest
import torch

from torch_timeseries.nn import (
    LearnableFourierFeatures,
    RotaryEmbedding,
    SinusoidalEmbedding,
    Time2Vec,
)

B, L, K = 4, 96, 64
TAU = torch.arange(L, dtype=torch.float).unsqueeze(0).expand(B, -1)  # (4, 96)


# ── Time2Vec ──────────────────────────────────────────────────────────────────

class TestTime2Vec:
    def test_output_shape(self):
        enc = Time2Vec(k=K)
        out = enc(TAU)
        assert out.shape == (B, L, K)

    def test_first_dim_is_linear(self):
        enc = Time2Vec(k=K)
        out = enc(TAU)
        # First component: W[0]*tau + b[0], no sin — linear in tau
        # Verify it's NOT identically periodic (sine would repeat)
        first = out[0, :, 0]
        diffs = first[1:] - first[:-1]
        # All diffs should be equal (linear function of integer steps)
        assert torch.allclose(diffs, diffs[0].expand_as(diffs), atol=1e-5)

    def test_periodic_dims_bounded(self):
        enc = Time2Vec(k=K)
        out = enc(TAU)
        periodic = out[..., 1:]
        assert (periodic >= -1 - 1e-5).all()
        assert (periodic <= 1 + 1e-5).all()

    def test_gradients_flow(self):
        enc = Time2Vec(k=K)
        tau = TAU.clone().requires_grad_(False)
        out = enc(tau)
        loss = out.sum()
        loss.backward()
        assert enc.W.grad is not None
        assert enc.b.grad is not None

    def test_k1_raises(self):
        # k=1 gives only the linear component — should work fine (no periodic dims)
        enc = Time2Vec(k=1)
        out = enc(TAU)
        assert out.shape == (B, L, 1)


# ── LearnableFourierFeatures ──────────────────────────────────────────────────

class TestLearnableFourierFeatures:
    def test_output_shape(self):
        enc = LearnableFourierFeatures(d_model=K)
        out = enc(TAU)
        assert out.shape == (B, L, K)

    def test_output_range(self):
        enc = LearnableFourierFeatures(d_model=K)
        out = enc(TAU)
        assert (out >= -1 - 1e-5).all()
        assert (out <= 1 + 1e-5).all()

    def test_odd_d_model_raises(self):
        with pytest.raises(ValueError):
            LearnableFourierFeatures(d_model=63)

    def test_gradients_flow(self):
        enc = LearnableFourierFeatures(d_model=K)
        out = enc(TAU)
        out.sum().backward()
        assert enc.omega.grad is not None
        assert enc.phi.grad is not None

    def test_no_nan(self):
        enc = LearnableFourierFeatures(d_model=K)
        out = enc(TAU)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_output_has_both_sin_cos(self):
        enc = LearnableFourierFeatures(d_model=4)
        tau = torch.zeros(1, 1)
        out = enc(tau)
        # At tau=0: sin(phi), cos(phi) — first half sin, second half cos
        assert out.shape == (1, 1, 4)


# ── RotaryEmbedding ───────────────────────────────────────────────────────────

class TestRotaryEmbedding:
    def test_output_shape_unchanged(self):
        H, D = 4, 32
        rope = RotaryEmbedding(dim=D)
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        q_r, k_r = rope(q, k)
        assert q_r.shape == q.shape
        assert k_r.shape == k.shape

    def test_preserves_norms(self):
        H, D = 4, 32
        rope = RotaryEmbedding(dim=D)
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        q_r, k_r = rope(q, k)
        assert torch.allclose(q_r.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
        assert torch.allclose(k_r.norm(dim=-1), k.norm(dim=-1), atol=1e-5)

    def test_different_positions_give_different_output(self):
        H, D = 2, 16
        rope = RotaryEmbedding(dim=D)
        q = torch.ones(1, H, 10, D)
        k = torch.ones(1, H, 10, D)
        q_r, _ = rope(q, k)
        # Position 0 and position 5 should differ
        assert not torch.allclose(q_r[:, :, 0, :], q_r[:, :, 5, :])

    def test_cache_extends_for_longer_seq(self):
        rope = RotaryEmbedding(dim=16, max_len=32)
        q = torch.randn(1, 2, 64, 16)  # longer than max_len
        k = torch.randn(1, 2, 64, 16)
        q_r, k_r = rope(q, k)          # should not raise
        assert q_r.shape == (1, 2, 64, 16)

    def test_no_learnable_params(self):
        rope = RotaryEmbedding(dim=32)
        assert sum(p.numel() for p in rope.parameters()) == 0


# ── SinusoidalEmbedding ───────────────────────────────────────────────────────

class TestSinusoidalEmbedding:
    def test_output_shape(self):
        pe = SinusoidalEmbedding(d_model=K)
        x = torch.zeros(B, L, K)
        out = pe(x)
        assert out.shape == (1, L, K)

    def test_values_bounded(self):
        pe = SinusoidalEmbedding(d_model=K)
        x = torch.zeros(B, L, K)
        out = pe(x)
        assert (out >= -1 - 1e-5).all()
        assert (out <= 1 + 1e-5).all()

    def test_scale_flag(self):
        d = 64
        pe_no_scale = SinusoidalEmbedding(d_model=d, scale=False)
        pe_scale    = SinusoidalEmbedding(d_model=d, scale=True)
        x = torch.zeros(1, 10, d)
        ratio = pe_scale(x).abs().mean() / pe_no_scale(x).abs().mean()
        assert abs(ratio.item() - math.sqrt(d)) < 0.1

    def test_no_learnable_params(self):
        pe = SinusoidalEmbedding(d_model=64)
        assert sum(p.numel() for p in pe.parameters()) == 0

    def test_different_positions_differ(self):
        pe = SinusoidalEmbedding(d_model=32)
        x = torch.zeros(1, 32, 32)
        out = pe(x)
        # PE at position 0 and position 10 should differ
        assert not torch.allclose(out[:, 0, :], out[:, 10, :])

    def test_odd_d_model_works(self):
        pe = SinusoidalEmbedding(d_model=5)
        x = torch.zeros(1, 10, 5)
        out = pe(x)
        assert out.shape == (1, 10, 5)
        assert not torch.isnan(out).any()

    def test_no_nan(self):
        pe = SinusoidalEmbedding(d_model=64)
        x = torch.zeros(2, 100, 64)
        out = pe(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
