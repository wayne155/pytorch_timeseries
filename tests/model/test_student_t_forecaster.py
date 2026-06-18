"""Tests for StudentTForecaster — heavy-tailed distributional head."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.StudentTForecaster import StudentTForecaster
from torch_timeseries.model.VanillaTransformer import VanillaTransformer


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_backbone(seq_len=24, pred_len=12, enc_in=7):
    return VanillaTransformer(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, dropout=0.1
    )


def _make_model(enc_in=7, pred_len=12, num_samples=10):
    return StudentTForecaster(
        backbone=_make_backbone(pred_len=pred_len, enc_in=enc_in),
        enc_in=enc_in,
        pred_len=pred_len,
        num_samples=num_samples,
    )


# ── construction ──────────────────────────────────────────────────────────────


class TestStudentTConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_num_samples_stored(self):
        assert _make_model(num_samples=25).num_samples == 25

    def test_default_num_samples(self):
        m = StudentTForecaster(_make_backbone(), enc_in=7, pred_len=12)
        assert m.num_samples == 50

    def test_two_extra_heads(self):
        m = _make_model()
        assert isinstance(m.log_sigma_head, nn.Linear)
        assert isinstance(m.log_nu_head, nn.Linear)

    def test_nu_bias_init(self):
        """log_nu bias initialised to 2.0 → ν ≈ e² ≈ 7."""
        m = _make_model()
        assert torch.allclose(
            m.log_nu_head.bias,
            torch.full_like(m.log_nu_head.bias, 2.0),
            atol=1e-5,
        )

    def test_clamp_params_stored(self):
        m = StudentTForecaster(
            _make_backbone(), enc_in=7, pred_len=12,
            min_log_nu=1.0, max_log_nu=3.0,
        )
        assert m.min_log_nu == 1.0
        assert m.max_log_nu == 3.0


# ── forward: (mu, log_sigma, log_nu) ─────────────────────────────────────────


class TestStudentTForward:
    def test_forward_returns_three_tensors(self):
        m = _make_model()
        out = m(torch.randn(4, 24, 7))
        assert isinstance(out, tuple) and len(out) == 3

    def test_shapes(self):
        m = _make_model(enc_in=7, pred_len=12)
        mu, log_sigma, log_nu = m(torch.randn(4, 24, 7))
        for t in (mu, log_sigma, log_nu):
            assert t.shape == (4, 12, 7)

    def test_log_nu_clamped(self):
        m = StudentTForecaster(
            _make_backbone(), enc_in=7, pred_len=12,
            min_log_nu=1.0, max_log_nu=2.0,
        )
        _, _, log_nu = m(torch.randn(2, 24, 7))
        assert (log_nu >= 1.0).all() and (log_nu <= 2.0).all()

    def test_log_sigma_clamped(self):
        m = StudentTForecaster(
            _make_backbone(), enc_in=7, pred_len=12,
            min_log_sigma=-3.0, max_log_sigma=1.0,
        )
        _, log_sigma, _ = m(torch.randn(2, 24, 7))
        assert (log_sigma >= -3.0).all() and (log_sigma <= 1.0).all()

    def test_output_finite(self):
        m = _make_model()
        mu, log_sigma, log_nu = m(torch.randn(4, 24, 7))
        for t in (mu, log_sigma, log_nu):
            assert torch.isfinite(t).all()

    def test_gradient_flows(self):
        m = _make_model()
        x = torch.randn(2, 24, 7)
        mu, log_sigma, log_nu = m(x)
        (mu + log_sigma + log_nu).sum().backward()
        has_grad = any(p.grad is not None for p in m.parameters())
        assert has_grad


# ── nll_loss ──────────────────────────────────────────────────────────────────


class TestStudentTNLL:
    def test_returns_scalar(self):
        m = _make_model()
        loss = m.nll_loss(torch.randn(4, 24, 7), torch.randn(4, 12, 7))
        assert loss.ndim == 0

    def test_finite(self):
        m = _make_model()
        loss = m.nll_loss(torch.randn(4, 24, 7), torch.randn(4, 12, 7))
        assert torch.isfinite(loss)

    def test_gradient_through_loss(self):
        m = _make_model()
        loss = m.nll_loss(torch.randn(2, 24, 7), torch.randn(2, 12, 7))
        loss.backward()
        assert any(p.grad is not None for p in m.parameters())

    def test_heavy_tail_differs_from_gaussian(self):
        """NLL for off-mean samples should differ between heavy and light tails."""
        backbone = _make_backbone()
        m_heavy = StudentTForecaster(
            backbone, enc_in=7, pred_len=12,
            min_log_nu=0.69, max_log_nu=0.70,  # ν≈2 — very heavy tails
        )
        m_light = StudentTForecaster(
            backbone, enc_in=7, pred_len=12,
            min_log_nu=3.4, max_log_nu=3.5,   # ν≈30 — near-Gaussian
        )
        torch.manual_seed(0)
        x = torch.zeros(4, 24, 7)
        y = torch.full((4, 12, 7), 5.0)   # far from mean
        # Heavy tails should give lower penalty for outliers
        loss_heavy = m_heavy.nll_loss(x, y).item()
        loss_light = m_light.nll_loss(x, y).item()
        assert loss_heavy < loss_light


# ── sample ────────────────────────────────────────────────────────────────────


class TestStudentTSample:
    def test_sample_shape(self):
        m = _make_model(num_samples=8)
        samples = m.sample(torch.randn(4, 24, 7))
        assert samples.shape == (4, 12, 7, 8)

    def test_sample_override(self):
        m = _make_model(num_samples=10)
        samples = m.sample(torch.randn(2, 24, 7), num_samples=3)
        assert samples.shape == (2, 12, 7, 3)

    def test_samples_no_grad(self):
        m = _make_model()
        assert not m.sample(torch.randn(2, 24, 7)).requires_grad

    def test_samples_finite(self):
        m = _make_model(num_samples=5)
        assert torch.isfinite(m.sample(torch.randn(2, 24, 7))).all()

    def test_samples_stochastic(self):
        m = _make_model(num_samples=20)
        samples = m.sample(torch.randn(1, 24, 7))
        assert not torch.allclose(samples[..., 0], samples[..., 1])


# ── registry ──────────────────────────────────────────────────────────────────


class TestStudentTRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import StudentTForecaster as S
        assert S is StudentTForecaster

    def test_experiment_importable(self):
        from torch_timeseries.experiments.StudentTForecaster import StudentTForecast
        assert StudentTForecast.model_type == "StudentT"

    def test_in_experiment_registry(self):
        from torch_timeseries.experiments import get_experiment_class
        cls = get_experiment_class("StudentT", "Forecast")
        assert cls.__name__ == "StudentTForecast"
