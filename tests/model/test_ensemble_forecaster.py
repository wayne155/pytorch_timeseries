"""Tests for EnsembleForecaster — Deep Ensemble probabilistic wrapper."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.EnsembleForecaster import EnsembleForecaster
from torch_timeseries.model.VanillaTransformer import VanillaTransformer


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_backbone_fn(seq_len=24, pred_len=12, enc_in=7):
    def fn():
        return VanillaTransformer(
            seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, dropout=0.1
        )
    return fn


def _make_model(num_members=3, enc_in=7, pred_len=12):
    return EnsembleForecaster(
        backbone_fn=_make_backbone_fn(pred_len=pred_len, enc_in=enc_in),
        num_members=num_members,
    )


# ── construction ──────────────────────────────────────────────────────────────


class TestEnsembleConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_num_members_stored(self):
        m = _make_model(num_members=7)
        assert m.num_members == 7

    def test_members_is_module_list(self):
        m = _make_model(num_members=3)
        assert isinstance(m.members, nn.ModuleList)
        assert len(m.members) == 3

    def test_members_are_distinct_modules(self):
        """Each backbone should be a separate nn.Module (not shared)."""
        m = _make_model(num_members=2)
        assert m.members[0] is not m.members[1]

    def test_members_have_independent_weights(self):
        """Different random seeds → different initial weights on learned layers."""
        m = _make_model(num_members=2)
        # Skip bias/affine parameters (may share deterministic init); compare learned weights
        params0 = [p for name, p in m.members[0].named_parameters() if "weight" in name and p.dim() >= 2]
        params1 = [p for name, p in m.members[1].named_parameters() if "weight" in name and p.dim() >= 2]
        assert len(params0) > 0
        any_differ = any(not torch.allclose(p0, p1) for p0, p1 in zip(params0, params1))
        assert any_differ, "All 2D weight matrices are identical — members may share parameters"

    def test_rejects_zero_members(self):
        with pytest.raises(ValueError, match="num_members"):
            EnsembleForecaster(backbone_fn=_make_backbone_fn(), num_members=0)

    def test_single_member_allowed(self):
        m = EnsembleForecaster(backbone_fn=_make_backbone_fn(), num_members=1)
        assert m.num_members == 1


# ── forward ───────────────────────────────────────────────────────────────────


class TestEnsembleForward:
    def test_forward_shape(self):
        m = _make_model(num_members=3)
        out = m(torch.randn(4, 24, 7))
        assert out.shape == (4, 12, 7)

    def test_forward_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 24, 7))).all()

    def test_forward_is_mean_of_members(self):
        m = _make_model(num_members=3).eval()
        x = torch.randn(2, 24, 7)
        with torch.no_grad():
            preds = torch.stack([mem(x) for mem in m.members], dim=-1)
            expected = preds.mean(dim=-1)
            actual = m(x)
        assert torch.allclose(actual, expected)

    def test_gradient_flows(self):
        m = _make_model(num_members=2)
        out = m(torch.randn(2, 24, 7))
        out.sum().backward()
        # All member parameters should get gradients
        for member in m.members:
            assert any(p.grad is not None for p in member.parameters())


# ── mse_loss ──────────────────────────────────────────────────────────────────


class TestEnsembleMSELoss:
    def test_returns_scalar(self):
        m = _make_model()
        loss = m.mse_loss(torch.randn(4, 24, 7), torch.randn(4, 12, 7))
        assert loss.ndim == 0

    def test_non_negative(self):
        m = _make_model()
        loss = m.mse_loss(torch.randn(4, 24, 7), torch.randn(4, 12, 7))
        assert loss.item() >= 0.0

    def test_finite(self):
        m = _make_model()
        assert torch.isfinite(m.mse_loss(torch.randn(2, 24, 7), torch.randn(2, 12, 7)))

    def test_gradient_through_loss(self):
        m = _make_model(num_members=2)
        loss = m.mse_loss(torch.randn(2, 24, 7), torch.randn(2, 12, 7))
        loss.backward()
        for member in m.members:
            assert any(p.grad is not None for p in member.parameters())

    def test_perfect_prediction_near_zero(self):
        m = _make_model(num_members=1)
        with torch.no_grad():
            for p in m.parameters():
                p.zero_()
        x = torch.zeros(2, 24, 7)
        y = torch.zeros(2, 12, 7)
        loss = m.mse_loss(x, y)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)


# ── sample ────────────────────────────────────────────────────────────────────


class TestEnsembleSample:
    def test_sample_shape(self):
        m = _make_model(num_members=5)
        samples = m.sample(torch.randn(4, 24, 7))
        assert samples.shape == (4, 12, 7, 5)

    def test_sample_override_members(self):
        m = _make_model(num_members=5)
        samples = m.sample(torch.randn(2, 24, 7), num_samples=3)
        assert samples.shape == (2, 12, 7, 3)

    def test_samples_no_grad(self):
        m = _make_model()
        assert not m.sample(torch.randn(2, 24, 7)).requires_grad

    def test_samples_finite(self):
        m = _make_model(num_members=3)
        assert torch.isfinite(m.sample(torch.randn(2, 24, 7))).all()

    def test_members_produce_different_outputs(self):
        """Independently initialised members should give different predictions."""
        m = _make_model(num_members=3)
        m.eval()
        samples = m.sample(torch.randn(1, 24, 7))  # (1, 12, 7, 3)
        assert not torch.allclose(samples[..., 0], samples[..., 1])

    def test_num_samples_clipped_to_num_members(self):
        """Requesting more samples than members should silently cap."""
        m = _make_model(num_members=3)
        samples = m.sample(torch.randn(2, 24, 7), num_samples=100)
        assert samples.shape[-1] == 3


# ── registry ──────────────────────────────────────────────────────────────────


class TestEnsembleRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import EnsembleForecaster as E
        assert E is EnsembleForecaster

    def test_experiment_importable(self):
        from torch_timeseries.experiments.EnsembleForecaster import EnsembleForecast
        assert EnsembleForecast.model_type == "Ensemble"

    def test_in_experiment_registry(self):
        from torch_timeseries.experiments import get_experiment_class
        cls = get_experiment_class("Ensemble", "Forecast")
        assert cls.__name__ == "EnsembleForecast"
