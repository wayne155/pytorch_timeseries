"""Tests for torch_timeseries.nn.RevIN."""
import pytest
import torch

from torch_timeseries.nn import RevIN

B, T, C = 4, 96, 7


# ── basic shape ───────────────────────────────────────────────────────────────

def test_output_shape_norm():
    m = RevIN(C)
    x = torch.randn(B, T, C)
    out = m(x, mode="norm")
    assert out.shape == x.shape


def test_output_shape_denorm():
    m = RevIN(C)
    x = torch.randn(B, T, C)
    m(x, mode="norm")
    out = m(x, mode="denorm")
    assert out.shape == x.shape


# ── normalization correctness ────────────────────────────────────────────────

def test_norm_mean_near_zero():
    m = RevIN(C, affine=False)
    x = torch.randn(B, T, C) + 5        # large offset
    x_n = m(x, mode="norm")
    # per-instance mean along time dim should be ~0
    assert x_n.mean(dim=1).abs().max() < 1e-5


def test_norm_std_near_one():
    m = RevIN(C, affine=False)
    x = torch.randn(B, T, C) * 10
    x_n = m(x, mode="norm")
    # per-instance std along time dim should be ~1
    # (biased std = sqrt(var), so compare against that)
    std = x_n.var(dim=1, unbiased=False).sqrt()
    assert (std - 1).abs().max() < 1e-4


# ── roundtrip ────────────────────────────────────────────────────────────────

def test_roundtrip_no_affine():
    m = RevIN(C, affine=False)
    x = torch.randn(B, T, C)
    x_back = m(m(x, mode="norm"), mode="denorm")
    assert torch.allclose(x_back, x, atol=1e-5)


def test_roundtrip_affine_identity_init():
    # affine starts as weight=1, bias=0 so roundtrip should hold
    m = RevIN(C, affine=True)
    x = torch.randn(B, T, C)
    x_back = m(m(x, mode="norm"), mode="denorm")
    assert torch.allclose(x_back, x, atol=1e-4)


# ── learnable parameters ─────────────────────────────────────────────────────

def test_affine_parameters_exist():
    m = RevIN(C, affine=True)
    names = [n for n, _ in m.named_parameters()]
    assert "affine_weight" in names
    assert "affine_bias" in names


def test_no_affine_no_parameters():
    m = RevIN(C, affine=False)
    assert sum(1 for _ in m.parameters()) == 0


# ── gradient flow ────────────────────────────────────────────────────────────

def test_grad_flows_through_norm():
    m = RevIN(C, affine=True)
    x = torch.randn(B, T, C, requires_grad=True)
    out = m(x, mode="norm")
    out.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_affine_params_get_grads():
    m = RevIN(C, affine=True)
    x = torch.randn(B, T, C)
    out = m(x, mode="norm")
    out.sum().backward()
    assert m.affine_weight.grad is not None
    assert m.affine_bias.grad is not None


# ── denorm before norm raises ────────────────────────────────────────────────

def test_denorm_before_norm_raises():
    m = RevIN(C)
    x = torch.randn(B, T, C)
    with pytest.raises(RuntimeError, match="norm"):
        m(x, mode="denorm")


def test_invalid_mode_raises():
    m = RevIN(C)
    with pytest.raises(ValueError, match="mode"):
        m(torch.randn(B, T, C), mode="invalid")


# ── forecast-like use pattern ────────────────────────────────────────────────

def test_forecaster_pipeline():
    """RevIN normalises lookback, model predicts, RevIN denorms forecast."""
    H = 24
    m = RevIN(C, affine=True)
    x = torch.randn(B, T, C) * 10 + 100   # scaled, shifted

    x_norm = m(x, mode="norm")
    # simulate model: project last step to forecast horizon
    fake_pred = x_norm[:, -1:, :].expand(B, H, C)
    pred_orig = m(fake_pred, mode="denorm")

    # result should be on the same scale as input, not ~0
    assert pred_orig.abs().mean() > 1.0
