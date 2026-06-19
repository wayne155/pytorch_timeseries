"""Tests for NormalizingFlowForecaster — Real-NVP conditional flow model."""
import math
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.NormalizingFlowForecaster import (
    NormalizingFlowForecaster,
    _AffineCouplingBlock,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_model(seq_len=24, pred_len=8, enc_in=3, d_model=32, n_heads=2,
                e_layers=1, d_ff=64, flow_layers=2, flow_hidden=32,
                num_samples=5, revin=True):
    return NormalizingFlowForecaster(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff,
        dropout=0.0, activation="gelu", revin=revin,
        flow_layers=flow_layers, flow_hidden=flow_hidden, num_samples=num_samples,
    )


# ── AffineCouplingBlock ────────────────────────────────────────────────────────


class TestAffineCouplingBlock:
    def test_forward_output_shape(self):
        block = _AffineCouplingBlock(dim=16, ctx_dim=16, hidden=32)
        z = torch.randn(4, 16)
        ctx = torch.randn(4, 16)
        z_out, log_det = block(z, ctx)
        assert z_out.shape == (4, 16)
        assert log_det.shape == (4,)

    def test_first_half_unchanged(self):
        block = _AffineCouplingBlock(dim=16, ctx_dim=16, hidden=32)
        z = torch.randn(4, 16)
        ctx = torch.randn(4, 16)
        z_out, _ = block(z, ctx)
        # First half should be exactly unchanged
        assert torch.allclose(z_out[:, :8], z[:, :8])

    def test_inverse_is_exact(self):
        block = _AffineCouplingBlock(dim=16, ctx_dim=16, hidden=32)
        z = torch.randn(4, 16)
        ctx = torch.randn(4, 16)
        z_fwd, _ = block(z, ctx)
        z_rec = block.inverse(z_fwd, ctx)
        assert torch.allclose(z, z_rec, atol=1e-5)

    def test_log_det_shape(self):
        block = _AffineCouplingBlock(dim=8, ctx_dim=4, hidden=16)
        z = torch.randn(3, 8)
        ctx = torch.randn(3, 4)
        _, log_det = block(z, ctx)
        assert log_det.shape == (3,)

    def test_identity_init(self):
        """At init, scale outputs ≈ 0 so log_det ≈ 0."""
        block = _AffineCouplingBlock(dim=16, ctx_dim=16, hidden=32)
        z = torch.randn(2, 16)
        ctx = torch.randn(2, 16)
        with torch.no_grad():
            _, log_det = block(z, ctx)
        assert log_det.abs().max().item() < 1e-5

    def test_gradient_flows(self):
        block = _AffineCouplingBlock(dim=16, ctx_dim=16, hidden=32)
        z = torch.randn(4, 16, requires_grad=True)
        ctx = torch.randn(4, 16)
        z_out, log_det = block(z, ctx)
        (z_out.sum() + log_det.sum()).backward()
        assert z.grad is not None


# ── NormalizingFlowForecaster ──────────────────────────────────────────────────


class TestNormalizingFlowConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_num_flow_layers(self):
        m = _make_model(flow_layers=4)
        assert len(m.flows) == 4

    def test_all_flows_are_coupling_blocks(self):
        m = _make_model()
        for f in m.flows:
            assert isinstance(f, _AffineCouplingBlock)

    def test_num_samples_stored(self):
        m = _make_model(num_samples=10)
        assert m.num_samples == 10


class TestNormalizingFlowNLLLoss:
    def test_nll_is_scalar(self):
        m = _make_model()
        x = torch.randn(4, 24, 3)
        y = torch.randn(4, 8, 3)
        loss = m.nll_loss(x, y)
        assert loss.ndim == 0

    def test_nll_is_finite(self):
        m = _make_model()
        x = torch.randn(4, 24, 3)
        y = torch.randn(4, 8, 3)
        loss = m.nll_loss(x, y)
        assert torch.isfinite(loss)

    def test_nll_backward(self):
        m = _make_model()
        x = torch.randn(2, 24, 3)
        y = torch.randn(2, 8, 3)
        loss = m.nll_loss(x, y)
        loss.backward()
        assert m.flows[0].s_net[0].weight.grad is not None

    def test_nll_decreases_with_training(self):
        torch.manual_seed(0)
        m = _make_model(flow_layers=2, e_layers=1)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        x = torch.randn(8, 24, 3)
        y = torch.randn(8, 8, 3)
        losses = []
        for _ in range(20):
            opt.zero_grad()
            loss = m.nll_loss(x, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], "NLL should decrease with training"


class TestNormalizingFlowForward:
    def test_forward_output_shape(self):
        m = _make_model()
        out = m(torch.randn(4, 24, 3))
        assert out.shape == (4, 8, 3)

    def test_forward_is_backbone(self):
        """forward() returns the backbone prediction (mean forecast)."""
        m = _make_model()
        x = torch.randn(2, 24, 3)
        with torch.no_grad():
            out_fwd = m(x)
            out_bb = m.backbone(x)
        assert torch.allclose(out_fwd, out_bb)

    def test_forward_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 24, 3))).all()


class TestNormalizingFlowSample:
    def test_sample_shape(self):
        m = _make_model(num_samples=5)
        out = m.sample(torch.randn(4, 24, 3))
        assert out.shape == (4, 8, 3, 5)

    def test_sample_custom_S(self):
        m = _make_model()
        out = m.sample(torch.randn(2, 24, 3), num_samples=7)
        assert out.shape == (2, 8, 3, 7)

    def test_sample_is_finite(self):
        m = _make_model()
        samples = m.sample(torch.randn(2, 24, 3))
        assert torch.isfinite(samples).all()

    def test_samples_are_diverse(self):
        """Samples from the flow should vary across the S dimension."""
        m = _make_model(num_samples=10)
        samples = m.sample(torch.randn(2, 24, 3))
        std_over_samples = samples.std(dim=-1)
        assert std_over_samples.mean().item() > 1e-5


class TestNormalizingFlowRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import NormalizingFlowForecaster as N
        assert N is NormalizingFlowForecaster

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.NormalizingFlowForecaster import NormalizingFlowForecast
        assert NormalizingFlowForecast.model_type == "NormalizingFlow"

    def test_registered_for_forecast(self):
        from torch_timeseries.experiments import get_experiment_class
        cls = get_experiment_class("NormalizingFlow", "Forecast")
        assert cls is not None

    def test_not_registered_for_other_tasks(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("AnomalyDetection", "Imputation", "UEAClassification"):
            with pytest.raises(NotImplementedError):
                get_experiment_class("NormalizingFlow", task)
