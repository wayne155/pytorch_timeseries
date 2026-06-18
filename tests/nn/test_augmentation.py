"""Tests for torch_timeseries.nn augmentation modules."""
import pytest
import torch

from torch_timeseries.nn import (
    Compose,
    Flip,
    Jitter,
    MagnitudeWarp,
    Scaling,
    TimeWarp,
    WindowCutout,
)

B, L, C = 4, 96, 7


def _x():
    return torch.randn(B, L, C)


# ── helpers ───────────────────────────────────────────────────────────────────

def _train(module):
    module.train()
    return module


def _eval(module):
    module.eval()
    return module


# ── Jitter ────────────────────────────────────────────────────────────────────

class TestJitter:
    def test_output_shape(self):
        x = _x()
        out = _train(Jitter(sigma=0.1))(x)
        assert out.shape == x.shape

    def test_identity_in_eval(self):
        m = _eval(Jitter(sigma=0.5))
        x = _x()
        assert torch.allclose(m(x), x)

    def test_output_differs_from_input_in_training(self):
        m = _train(Jitter(sigma=1.0))
        x = _x()
        assert not torch.allclose(m(x), x)

    def test_sigma_zero_is_identity_in_training(self):
        m = _train(Jitter(sigma=0.0))
        x = _x()
        assert torch.allclose(m(x), x)

    def test_no_nan(self):
        m = _train(Jitter(sigma=0.1))
        assert not torch.isnan(m(_x())).any()


# ── Scaling ───────────────────────────────────────────────────────────────────

class TestScaling:
    def test_output_shape(self):
        out = _train(Scaling(sigma=0.1))(_x())
        assert out.shape == (B, L, C)

    def test_identity_in_eval(self):
        m = _eval(Scaling(sigma=0.5))
        x = _x()
        assert torch.allclose(m(x), x)

    def test_scale_is_broadcast_over_time(self):
        m = _train(Scaling(sigma=2.0))
        x = torch.ones(B, L, C)
        out = m(x)
        # Each sample should have a constant ratio across time
        for i in range(B):
            ratios = out[i, :, 0]
            assert torch.allclose(ratios, ratios[0:1].expand_as(ratios), atol=1e-5)

    def test_no_nan(self):
        assert not torch.isnan(_train(Scaling(0.1))(_x())).any()


# ── Flip ──────────────────────────────────────────────────────────────────────

class TestFlip:
    def test_output_shape(self):
        out = _train(Flip(p=0.5))(_x())
        assert out.shape == (B, L, C)

    def test_identity_in_eval(self):
        m = _eval(Flip(p=1.0))
        x = _x()
        assert torch.allclose(m(x), x)

    def test_p_zero_never_flips(self):
        m = _train(Flip(p=0.0))
        x = _x()
        assert torch.allclose(m(x), x)

    def test_p_one_always_flips(self):
        m = _train(Flip(p=1.0))
        x = _x()
        out = m(x)
        assert torch.allclose(out, x.flip(dims=[1]))

    def test_no_nan(self):
        assert not torch.isnan(_train(Flip(p=0.5))(_x())).any()


# ── WindowCutout ──────────────────────────────────────────────────────────────

class TestWindowCutout:
    def test_output_shape(self):
        out = _train(WindowCutout(min_len=5, max_len=15, p=1.0))(_x())
        assert out.shape == (B, L, C)

    def test_identity_in_eval(self):
        m = _eval(WindowCutout(min_len=5, max_len=15, p=1.0))
        x = _x()
        assert torch.allclose(m(x), x)

    def test_p_zero_no_change(self):
        m = _train(WindowCutout(min_len=5, max_len=15, p=0.0))
        x = _x()
        assert torch.allclose(m(x), x)

    def test_output_contains_zeros(self):
        m = _train(WindowCutout(min_len=10, max_len=20, p=1.0))
        x = torch.ones(B, L, C)
        out = m(x)
        # At p=1.0, every sample should have some zeros
        for i in range(B):
            assert (out[i] == 0).any()

    def test_no_nan(self):
        assert not torch.isnan(_train(WindowCutout(5, 20, p=0.5))(_x())).any()


# ── MagnitudeWarp ─────────────────────────────────────────────────────────────

class TestMagnitudeWarp:
    def test_output_shape(self):
        out = _train(MagnitudeWarp(sigma=0.2, num_knots=4))(_x())
        assert out.shape == (B, L, C)

    def test_identity_in_eval(self):
        m = _eval(MagnitudeWarp(sigma=0.5))
        x = _x()
        assert torch.allclose(m(x), x)

    def test_sigma_zero_is_identity(self):
        m = _train(MagnitudeWarp(sigma=0.0, num_knots=4))
        x = _x()
        out = m(x)
        assert torch.allclose(out, x, atol=1e-5)

    def test_no_nan(self):
        assert not torch.isnan(_train(MagnitudeWarp(0.2, 4))(_x())).any()

    def test_constant_input_scaled(self):
        m = _train(MagnitudeWarp(sigma=1.0, num_knots=4))
        x = torch.ones(B, L, C)
        out = m(x)
        # output should equal the warp curve — no negative values for low sigma
        assert out.shape == (B, L, C)


# ── TimeWarp ──────────────────────────────────────────────────────────────────

class TestTimeWarp:
    def test_output_shape(self):
        out = _train(TimeWarp(sigma=0.2, num_knots=4))(_x())
        assert out.shape == (B, L, C)

    def test_identity_in_eval(self):
        m = _eval(TimeWarp(sigma=0.2))
        x = _x()
        assert torch.allclose(m(x), x)

    def test_no_nan(self):
        assert not torch.isnan(_train(TimeWarp(0.2, 4))(_x())).any()

    def test_constant_input_unchanged_value(self):
        m = _train(TimeWarp(sigma=0.5, num_knots=4))
        x = torch.ones(B, L, C) * 3.14
        out = m(x)
        assert torch.allclose(out, x, atol=1e-4)


# ── Compose ───────────────────────────────────────────────────────────────────

class TestCompose:
    def test_output_shape(self):
        aug = _train(Compose([Jitter(0.05), Scaling(0.1)]))
        out = aug(_x())
        assert out.shape == (B, L, C)

    def test_three_transforms(self):
        aug = _train(Compose([Jitter(0.05), Scaling(0.1), WindowCutout(5, 15, p=0.5)]))
        out = aug(_x())
        assert out.shape == (B, L, C)

    def test_identity_in_eval(self):
        aug = Compose([Jitter(0.5), Scaling(0.5), Flip(p=1.0)])
        aug.eval()
        x = _x()
        assert torch.allclose(aug(x), x)

    def test_empty_compose_is_identity(self):
        aug = _train(Compose([]))
        x = _x()
        assert torch.allclose(aug(x), x)

    def test_no_nan(self):
        aug = _train(Compose([
            Jitter(0.1), Scaling(0.1), Flip(0.5),
            MagnitudeWarp(0.2, 4), WindowCutout(5, 20, 0.5),
        ]))
        assert not torch.isnan(aug(_x())).any()
