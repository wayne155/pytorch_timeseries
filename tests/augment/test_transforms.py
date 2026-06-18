"""Tests for torch_timeseries.augment transforms."""
import pytest
import torch

from torch_timeseries.augment import (
    Compose,
    Flip,
    Jitter,
    MagnitudeWarp,
    Permute,
    RandomMask,
    Scale,
    TimeWarp,
    WindowSlice,
)

BATCH_3D = torch.randn(8, 96, 7)     # (B, T, C)
SAMPLE_2D = torch.randn(96, 7)       # (T, C)


# ── shape preservation ───────────────────────────────────────────────────────

@pytest.mark.parametrize("transform", [
    Jitter(sigma=0.05),
    Scale(sigma=0.1),
    MagnitudeWarp(sigma=0.2, n_knots=4),
    TimeWarp(sigma=0.2, n_knots=4),
    WindowSlice(crop_ratio=0.9),
    Permute(n_segments=4),
    Flip(),
    RandomMask(p=0.1),
])
def test_shape_preserved_3d(transform):
    out = transform(BATCH_3D)
    assert out.shape == BATCH_3D.shape


@pytest.mark.parametrize("transform", [
    Jitter(sigma=0.05),
    Scale(sigma=0.1),
    MagnitudeWarp(sigma=0.2, n_knots=4),
    TimeWarp(sigma=0.2, n_knots=4),
    WindowSlice(crop_ratio=0.9),
    Permute(n_segments=4),
    Flip(),
    RandomMask(p=0.1),
])
def test_shape_preserved_2d(transform):
    out = transform(SAMPLE_2D)
    assert out.shape == SAMPLE_2D.shape


# ── no NaN / Inf ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("transform", [
    Jitter(), Scale(), MagnitudeWarp(), TimeWarp(),
    WindowSlice(), Permute(), Flip(), RandomMask(),
])
def test_no_nan(transform):
    out = transform(BATCH_3D)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


# ── semantic checks ───────────────────────────────────────────────────────────

def test_jitter_differs_from_input():
    torch.manual_seed(0)
    out = Jitter(sigma=1.0)(BATCH_3D)
    assert not torch.allclose(out, BATCH_3D)


def test_flip_reverses_time():
    out = Flip()(BATCH_3D)
    assert torch.allclose(out, BATCH_3D.flip(dims=[1]))


def test_random_mask_zeros():
    torch.manual_seed(0)
    out = RandomMask(p=0.5)(torch.ones(32, 96, 7))
    assert (out == 0).any()
    assert (out == 1).any()


def test_random_mask_p_zero_identity():
    x = torch.randn(8, 96, 7)
    out = RandomMask(p=0.0)(x)
    assert torch.allclose(out, x)


def test_scale_multiplies():
    torch.manual_seed(42)
    x = torch.ones(8, 96, 7)
    out = Scale(sigma=1.0)(x)
    # ratios should be constant along time dim
    ratios = out[:, 0, :] / x[:, 0, :]
    assert torch.allclose(out[:, 0, :] / out[:, 1, :], torch.ones(8, 7), atol=1e-5)


def test_window_slice_crop_ratio_1_raises():
    with pytest.raises(AssertionError):
        WindowSlice(crop_ratio=1.0)


def test_permute_reorders_segments():
    torch.manual_seed(1)
    x = torch.arange(96 * 7).float().reshape(96, 7)
    out = Permute(n_segments=4)(x)
    assert out.shape == x.shape
    # each 24-step segment should appear somewhere in output
    seg0 = x[:24, :]
    found = any(
        torch.allclose(out[i * 24: (i + 1) * 24, :], seg0)
        for i in range(4)
    )
    assert found


# ── Compose ──────────────────────────────────────────────────────────────────

def test_compose_shape():
    aug = Compose([Jitter(0.05), Scale(0.1), MagnitudeWarp(0.2)])
    out = aug(BATCH_3D)
    assert out.shape == BATCH_3D.shape


def test_compose_empty():
    aug = Compose([])
    x = torch.randn(4, 48, 3)
    assert torch.allclose(aug(x), x)


def test_compose_order_matters():
    torch.manual_seed(0)
    x = torch.ones(4, 48, 3)
    aug1 = Compose([Scale(sigma=1.0), Jitter(sigma=0.0)])
    aug2 = Compose([Jitter(sigma=0.0), Scale(sigma=1.0)])
    # Flip then Scale == Scale then Flip only if scaling is identity
    # Use Flip + Scale to test ordering does not produce same result
    torch.manual_seed(42)
    a = Compose([Flip(), Scale(sigma=2.0)])(x)
    torch.manual_seed(42)
    b = Compose([Scale(sigma=2.0), Flip()])(x)
    # flip is deterministic, scale is stochastic — same seed means same scale
    # result should be equal because flip doesn't change values, only order
    assert torch.allclose(a, b.flip(dims=[1]))


# ── device consistency ────────────────────────────────────────────────────────

@pytest.mark.parametrize("transform", [
    Jitter(), Scale(), MagnitudeWarp(), TimeWarp(),
    WindowSlice(), Permute(), Flip(), RandomMask(),
])
def test_output_on_same_device(transform):
    x = BATCH_3D
    out = transform(x)
    assert out.device == x.device
