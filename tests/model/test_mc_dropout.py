"""Tests for MCDropoutForecaster."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.MCDropoutForecaster import MCDropoutForecaster
from torch_timeseries.model.VanillaTransformer import VanillaTransformer


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_backbone(seq_len=24, pred_len=12, enc_in=7, dropout=0.3):
    return VanillaTransformer(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, dropout=dropout
    )


def _make_model(num_samples=10, **kw):
    return MCDropoutForecaster(_make_backbone(**kw), num_samples=num_samples)


# ── construction ──────────────────────────────────────────────────────────────


class TestMCDropoutConstruction:
    def test_num_samples_stored(self):
        m = _make_model(num_samples=25)
        assert m.num_samples == 25

    def test_backbone_attribute(self):
        backbone = _make_backbone()
        m = MCDropoutForecaster(backbone)
        assert m.backbone is backbone

    def test_default_num_samples(self):
        m = MCDropoutForecaster(_make_backbone())
        assert m.num_samples == 50

    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_parameters_from_backbone(self):
        backbone = _make_backbone()
        m = MCDropoutForecaster(backbone)
        assert list(m.parameters()) == list(backbone.parameters())


# ── point forward (training) ──────────────────────────────────────────────────


class TestMCDropoutPointForward:
    def test_output_shape(self):
        m = _make_model()
        x = torch.randn(4, 24, 7)
        out = m(x)
        assert out.shape == (4, 12, 7)

    def test_gradient_flows(self):
        m = _make_model()
        m.train()
        x = torch.randn(2, 24, 7)
        loss = m(x).sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in m.parameters())
        assert has_grad

    def test_deterministic_in_eval_mode(self):
        m = _make_model()
        m.eval()
        x = torch.randn(2, 24, 7)
        with torch.no_grad():
            out1 = m(x)
            out2 = m(x)
        assert torch.allclose(out1, out2)

    def test_batch_size_one(self):
        m = _make_model()
        x = torch.randn(1, 24, 7)
        out = m(x)
        assert out.shape == (1, 12, 7)


# ── sample (inference) ───────────────────────────────────────────────────────


class TestMCDropoutSample:
    def test_sample_shape_default(self):
        m = _make_model(num_samples=10)
        x = torch.randn(4, 24, 7)
        samples = m.sample(x)
        assert samples.shape == (4, 12, 7, 10)

    def test_sample_shape_override(self):
        m = _make_model(num_samples=10)
        x = torch.randn(4, 24, 7)
        samples = m.sample(x, num_samples=5)
        assert samples.shape == (4, 12, 7, 5)

    def test_samples_are_stochastic(self):
        """With dropout=0.3, samples should differ (probability ≈ 1)."""
        m = _make_model(num_samples=20, dropout=0.3)
        m.eval()  # outer eval state; backbone stays in train inside sample()
        x = torch.randn(1, 24, 7)
        samples = m.sample(x)
        assert not torch.allclose(samples[..., 0], samples[..., 1])

    def test_sample_no_grad_tensor(self):
        m = _make_model()
        x = torch.randn(2, 24, 7)
        samples = m.sample(x)
        assert not samples.requires_grad

    def test_backbone_restored_to_eval_after_sample(self):
        m = _make_model()
        m.eval()
        x = torch.randn(2, 24, 7)
        m.sample(x)
        assert not m.backbone.training

    def test_backbone_stays_train_after_sample_when_in_train(self):
        m = _make_model()
        m.train()
        x = torch.randn(2, 24, 7)
        m.sample(x)
        assert m.backbone.training

    def test_batch_size_one(self):
        m = _make_model(num_samples=3)
        x = torch.randn(1, 24, 7)
        samples = m.sample(x)
        assert samples.shape == (1, 12, 7, 3)

    def test_last_dim_is_sample_axis(self):
        m = _make_model(num_samples=7)
        x = torch.randn(2, 24, 7)
        samples = m.sample(x)
        assert samples.shape[-1] == 7


# ── generic backbone ──────────────────────────────────────────────────────────


class TestMCDropoutGenericBackbone:
    """MCDropoutForecaster should work with any single-arg nn.Module."""

    @staticmethod
    def _tiny_mlp(seq_len=24, pred_len=12, enc_in=7):
        class TinyMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Flatten(1),
                    nn.Linear(seq_len * enc_in, pred_len * enc_in),
                    nn.Dropout(0.2),
                )

            def forward(self, x):
                return self.net(x).view(x.size(0), pred_len, enc_in)

        return TinyMLP()

    def test_forward_shape(self):
        m = MCDropoutForecaster(self._tiny_mlp(), num_samples=4)
        assert m(torch.randn(3, 24, 7)).shape == (3, 12, 7)

    def test_sample_shape(self):
        m = MCDropoutForecaster(self._tiny_mlp(), num_samples=4)
        assert m.sample(torch.randn(3, 24, 7)).shape == (3, 12, 7, 4)


# ── registry / import ─────────────────────────────────────────────────────────


class TestMCDropoutRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import MCDropoutForecaster as M
        assert M is MCDropoutForecaster

    def test_experiment_importable(self):
        from torch_timeseries.experiments.MCDropoutForecaster import MCDropoutForecast
        assert MCDropoutForecast.model_type == "MCDropout"

    def test_in_experiment_registry(self):
        from torch_timeseries.experiments import get_experiment_class
        cls = get_experiment_class("MCDropout", "Forecast")
        assert cls.__name__ == "MCDropoutForecast"
