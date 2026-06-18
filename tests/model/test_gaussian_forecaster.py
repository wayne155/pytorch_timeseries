"""Tests for GaussianForecaster — heteroscedastic distributional head."""
import math

import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.GaussianForecaster import GaussianForecaster
from torch_timeseries.model.VanillaTransformer import VanillaTransformer


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_backbone(seq_len=24, pred_len=12, enc_in=7):
    return VanillaTransformer(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, dropout=0.1
    )


def _make_model(enc_in=7, pred_len=12, num_samples=10):
    backbone = _make_backbone(pred_len=pred_len, enc_in=enc_in)
    return GaussianForecaster(
        backbone=backbone, enc_in=enc_in, pred_len=pred_len,
        num_samples=num_samples,
    )


# ── construction ──────────────────────────────────────────────────────────────


class TestGaussianForecasterConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_num_samples_stored(self):
        m = _make_model(num_samples=25)
        assert m.num_samples == 25

    def test_default_num_samples(self):
        backbone = _make_backbone()
        m = GaussianForecaster(backbone=backbone, enc_in=7, pred_len=12)
        assert m.num_samples == 50

    def test_log_sigma_head_is_linear(self):
        m = _make_model()
        assert isinstance(m.log_sigma_head, nn.Linear)

    def test_log_sigma_head_init_bias(self):
        """Bias initialised to -1.0 so σ starts near e^{-1} ≈ 0.37."""
        m = _make_model()
        assert torch.allclose(
            m.log_sigma_head.bias,
            torch.full_like(m.log_sigma_head.bias, -1.0),
            atol=1e-5,
        )

    def test_clamp_params_stored(self):
        m = GaussianForecaster(
            _make_backbone(), enc_in=7, pred_len=12,
            min_log_sigma=-5.0, max_log_sigma=3.0,
        )
        assert m.min_log_sigma == -5.0
        assert m.max_log_sigma == 3.0


# ── forward: (mu, log_sigma) ─────────────────────────────────────────────────


class TestGaussianForward:
    def test_forward_returns_two_tensors(self):
        m = _make_model()
        x = torch.randn(4, 24, 7)
        out = m(x)
        assert isinstance(out, tuple) and len(out) == 2

    def test_mu_shape(self):
        m = _make_model(enc_in=7, pred_len=12)
        mu, _ = m(torch.randn(4, 24, 7))
        assert mu.shape == (4, 12, 7)

    def test_log_sigma_shape(self):
        m = _make_model(enc_in=7, pred_len=12)
        _, log_sigma = m(torch.randn(4, 24, 7))
        assert log_sigma.shape == (4, 12, 7)

    def test_log_sigma_clamped(self):
        m = GaussianForecaster(
            _make_backbone(), enc_in=7, pred_len=12,
            min_log_sigma=-3.0, max_log_sigma=1.0,
        )
        _, log_sigma = m(torch.randn(4, 24, 7))
        assert (log_sigma >= -3.0).all()
        assert (log_sigma <= 1.0).all()

    def test_output_finite(self):
        m = _make_model()
        mu, log_sigma = m(torch.randn(4, 24, 7))
        assert torch.isfinite(mu).all()
        assert torch.isfinite(log_sigma).all()

    def test_gradient_flows_to_backbone(self):
        m = _make_model()
        x = torch.randn(2, 24, 7)
        mu, log_sigma = m(x)
        (mu + log_sigma).sum().backward()
        has_grad = any(p.grad is not None for p in m.backbone.parameters())
        assert has_grad


# ── nll_loss ──────────────────────────────────────────────────────────────────


class TestGaussianNLLLoss:
    def test_returns_scalar(self):
        m = _make_model()
        x = torch.randn(4, 24, 7)
        y = torch.randn(4, 12, 7)
        loss = m.nll_loss(x, y)
        assert loss.ndim == 0

    def test_finite(self):
        m = _make_model()
        x, y = torch.randn(4, 24, 7), torch.randn(4, 12, 7)
        assert torch.isfinite(m.nll_loss(x, y))

    def test_gradient_through_loss(self):
        m = _make_model()
        x, y = torch.randn(2, 24, 7), torch.randn(2, 12, 7)
        loss = m.nll_loss(x, y)
        loss.backward()
        has_grad = any(p.grad is not None for p in m.parameters())
        assert has_grad

    def test_perfect_prediction_has_finite_loss(self):
        """When pred = target, NLL is finite (depends on σ, not zero)."""
        m = _make_model()
        m.eval()
        with torch.no_grad():
            x = torch.randn(2, 24, 7)
            mu, _ = m(x)
        loss = m.nll_loss(x, mu.detach())
        assert torch.isfinite(loss)

    def test_lower_loss_with_better_sigma_scaling(self):
        """Smaller σ near the true noise level ⟹ lower NLL than large σ."""
        backbone = _make_backbone()
        m_tight = GaussianForecaster(backbone, enc_in=7, pred_len=12,
                                     min_log_sigma=-5.0, max_log_sigma=-4.9)
        m_wide = GaussianForecaster(backbone, enc_in=7, pred_len=12,
                                    min_log_sigma=1.9, max_log_sigma=2.0)
        x = torch.zeros(4, 24, 7)
        y = torch.zeros(4, 12, 7)  # truth = 0; backbone outputs near-zero mean
        # Wide σ always costs more NLL when |y-μ| ≈ 0 (only the log(σ) term matters)
        assert m_tight.nll_loss(x, y) < m_wide.nll_loss(x, y)


# ── sample ────────────────────────────────────────────────────────────────────


class TestGaussianSample:
    def test_sample_shape(self):
        m = _make_model(num_samples=10)
        x = torch.randn(4, 24, 7)
        samples = m.sample(x)
        assert samples.shape == (4, 12, 7, 10)

    def test_sample_override(self):
        m = _make_model(num_samples=10)
        samples = m.sample(torch.randn(2, 24, 7), num_samples=5)
        assert samples.shape == (2, 12, 7, 5)

    def test_sample_no_grad(self):
        m = _make_model()
        x = torch.randn(2, 24, 7)
        samples = m.sample(x)
        assert not samples.requires_grad

    def test_samples_are_stochastic(self):
        """Samples should vary across draws (probability 1)."""
        m = _make_model(num_samples=20)
        x = torch.randn(1, 24, 7)
        samples = m.sample(x)
        assert not torch.allclose(samples[..., 0], samples[..., 1])

    def test_samples_finite(self):
        m = _make_model(num_samples=5)
        samples = m.sample(torch.randn(2, 24, 7))
        assert torch.isfinite(samples).all()

    def test_samples_mean_close_to_mu(self):
        """For large S, sample mean ≈ μ."""
        m = _make_model(num_samples=500)
        m.eval()
        x = torch.randn(2, 24, 7)
        with torch.no_grad():
            mu, _ = m(x)
        samples = m.sample(x, num_samples=500)
        sample_mean = samples.mean(dim=-1)
        # Allow 3-sigma tolerance (σ/√500)
        assert torch.allclose(sample_mean, mu, atol=0.2)


# ── registry ──────────────────────────────────────────────────────────────────


class TestGaussianRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import GaussianForecaster as G
        assert G is GaussianForecaster

    def test_experiment_importable(self):
        from torch_timeseries.experiments.GaussianForecaster import GaussianForecast
        assert GaussianForecast.model_type == "Gaussian"

    def test_in_experiment_registry(self):
        from torch_timeseries.experiments import get_experiment_class
        cls = get_experiment_class("Gaussian", "Forecast")
        assert cls.__name__ == "GaussianForecast"
