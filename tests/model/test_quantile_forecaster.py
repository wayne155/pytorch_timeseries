"""Tests for QuantileForecaster — pinball-trained distributional head."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.QuantileForecaster import QuantileForecaster, _DEFAULT_QUANTILES
from torch_timeseries.model.VanillaTransformer import VanillaTransformer


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_backbone(seq_len=24, pred_len=12, enc_in=7):
    return VanillaTransformer(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, dropout=0.1
    )


def _make_model(enc_in=7, pred_len=12, quantiles=None):
    return QuantileForecaster(
        backbone=_make_backbone(pred_len=pred_len, enc_in=enc_in),
        enc_in=enc_in,
        pred_len=pred_len,
        quantiles=quantiles,
    )


# ── construction ──────────────────────────────────────────────────────────────


class TestQuantileConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_default_quantiles(self):
        m = _make_model()
        assert m.quantiles == list(_DEFAULT_QUANTILES)
        assert len(m.quantiles) == 9

    def test_custom_quantiles(self):
        qs = [0.1, 0.5, 0.9]
        m = _make_model(quantiles=qs)
        assert m.quantiles == qs

    def test_num_quantiles_property(self):
        m = _make_model(quantiles=[0.25, 0.5, 0.75])
        assert m.num_quantiles == 3

    def test_quantile_head_is_linear(self):
        assert isinstance(_make_model().quantile_head, nn.Linear)

    def test_quantile_head_zero_bias_init(self):
        m = _make_model()
        assert torch.allclose(m.quantile_head.bias, torch.zeros_like(m.quantile_head.bias))

    def test_rejects_empty_quantiles(self):
        with pytest.raises(ValueError, match="non-empty"):
            QuantileForecaster(_make_backbone(), enc_in=7, pred_len=12, quantiles=[])

    def test_rejects_unordered_quantiles(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            QuantileForecaster(_make_backbone(), enc_in=7, pred_len=12, quantiles=[0.9, 0.1])

    def test_rejects_out_of_range_quantile(self):
        with pytest.raises(ValueError, match="\\(0, 1\\)"):
            QuantileForecaster(_make_backbone(), enc_in=7, pred_len=12, quantiles=[0.0, 0.5])

    def test_rejects_quantile_equal_one(self):
        with pytest.raises(ValueError, match="\\(0, 1\\)"):
            QuantileForecaster(_make_backbone(), enc_in=7, pred_len=12, quantiles=[0.5, 1.0])


# ── forward ───────────────────────────────────────────────────────────────────


class TestQuantileForward:
    def test_shape(self):
        m = _make_model(enc_in=7, pred_len=12)
        out = m(torch.randn(4, 24, 7))
        assert out.shape == (4, 12, 7, 9)

    def test_custom_k_shape(self):
        m = _make_model(enc_in=5, pred_len=8, quantiles=[0.1, 0.5, 0.9])
        out = m(torch.randn(3, 24, 5))
        assert out.shape == (3, 8, 5, 3)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 24, 7))).all()

    def test_gradient_flows(self):
        m = _make_model()
        out = m(torch.randn(2, 24, 7))
        out.sum().backward()
        assert any(p.grad is not None for p in m.parameters())


# ── pinball_loss ──────────────────────────────────────────────────────────────


class TestPinballLoss:
    def test_returns_scalar(self):
        m = _make_model()
        loss = m.pinball_loss(torch.randn(4, 24, 7), torch.randn(4, 12, 7))
        assert loss.ndim == 0

    def test_finite(self):
        m = _make_model()
        assert torch.isfinite(m.pinball_loss(torch.randn(2, 24, 7), torch.randn(2, 12, 7)))

    def test_non_negative(self):
        m = _make_model()
        loss = m.pinball_loss(torch.randn(4, 24, 7), torch.randn(4, 12, 7))
        assert loss.item() >= 0.0

    def test_gradient_through_loss(self):
        m = _make_model()
        loss = m.pinball_loss(torch.randn(2, 24, 7), torch.randn(2, 12, 7))
        loss.backward()
        assert any(p.grad is not None for p in m.parameters())

    def test_perfect_prediction_is_zero(self):
        """When all quantile heads predict the exact target, loss is 0."""
        m = _make_model(enc_in=1, pred_len=1, quantiles=[0.5])
        with torch.no_grad():
            m.quantile_head.weight.zero_()
            m.quantile_head.bias.zero_()
            for p in m.backbone.parameters():
                p.zero_()
        x = torch.zeros(2, 24, 1)
        y = torch.zeros(2, 1, 1)
        loss = m.pinball_loss(x, y)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_asymmetric_penalty(self):
        """High quantile should penalise under-prediction more than low quantile."""
        m_low = _make_model(enc_in=1, pred_len=1, quantiles=[0.1])
        m_high = _make_model(enc_in=1, pred_len=1, quantiles=[0.9])
        # Ensure both models predict 0
        for m in (m_low, m_high):
            with torch.no_grad():
                m.quantile_head.weight.zero_()
                m.quantile_head.bias.zero_()
                for p in m.backbone.parameters():
                    p.zero_()
        # target = +1 → error > 0 → pinball = q * error
        x = torch.zeros(4, 24, 1)
        y = torch.ones(4, 1, 1)
        loss_low = m_low.pinball_loss(x, y).item()   # q=0.1 → 0.1
        loss_high = m_high.pinball_loss(x, y).item() # q=0.9 → 0.9
        assert loss_high > loss_low


# ── sample ────────────────────────────────────────────────────────────────────


class TestQuantileSample:
    def test_sample_shape(self):
        m = _make_model()
        samples = m.sample(torch.randn(4, 24, 7))
        assert samples.shape == (4, 12, 7, 9)

    def test_custom_quantiles_shape(self):
        m = _make_model(quantiles=[0.1, 0.5, 0.9])
        samples = m.sample(torch.randn(2, 24, 7))
        assert samples.shape == (2, 12, 7, 3)

    def test_samples_no_grad(self):
        m = _make_model()
        assert not m.sample(torch.randn(2, 24, 7)).requires_grad

    def test_samples_sorted_monotone(self):
        """sorted quantiles should be non-decreasing along last dim."""
        m = _make_model()
        samples = m.sample(torch.randn(4, 24, 7))   # (B, O, N, K)
        diffs = samples[..., 1:] - samples[..., :-1]
        assert (diffs >= 0).all()

    def test_samples_finite(self):
        m = _make_model()
        assert torch.isfinite(m.sample(torch.randn(2, 24, 7))).all()


# ── registry ──────────────────────────────────────────────────────────────────


class TestQuantileRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import QuantileForecaster as Q
        assert Q is QuantileForecaster

    def test_experiment_importable(self):
        from torch_timeseries.experiments.QuantileForecaster import QuantileForecast
        assert QuantileForecast.model_type == "Quantile"

    def test_in_experiment_registry(self):
        from torch_timeseries.experiments import get_experiment_class
        cls = get_experiment_class("Quantile", "Forecast")
        assert cls.__name__ == "QuantileForecast"
