"""Tests for CycleNet — learnable periodic cycle buffer + residual backbone."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.CycleNet import CycleNet


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_model(seq_len=96, pred_len=24, enc_in=7, cycle_len=24,
                backbone="linear", d_model=64, revin=True, dropout=0.0):
    return CycleNet(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        cycle_len=cycle_len, backbone=backbone, d_model=d_model,
        revin=revin, dropout=dropout,
    )


# ── construction ──────────────────────────────────────────────────────────────


class TestCycleNetConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_cycle_buffer_shape(self):
        m = CycleNet(seq_len=96, pred_len=24, enc_in=7, cycle_len=24)
        assert m.cycle.shape == (24, 7)

    def test_cycle_buffer_initialised_zeros(self):
        m = CycleNet(seq_len=96, pred_len=24, enc_in=7, cycle_len=24)
        assert m.cycle.abs().max().item() == 0.0

    def test_cycle_buffer_is_parameter(self):
        m = _make_model()
        assert m.cycle in list(m.parameters())

    def test_linear_backbone(self):
        m = CycleNet(seq_len=96, pred_len=24, enc_in=7, backbone="linear")
        assert isinstance(m.backbone, nn.Linear)

    def test_mlp_backbone(self):
        m = CycleNet(seq_len=96, pred_len=24, enc_in=7, backbone="mlp", d_model=64)
        assert isinstance(m.backbone, nn.Sequential)

    def test_invalid_backbone_raises(self):
        with pytest.raises(ValueError, match="backbone"):
            CycleNet(seq_len=96, pred_len=24, enc_in=7, backbone="transformer")

    def test_revin_exists_when_enabled(self):
        m = _make_model(revin=True)
        assert hasattr(m, "revin_layer")

    def test_no_revin_when_disabled(self):
        m = _make_model(revin=False)
        assert not hasattr(m, "revin_layer")

    def test_linear_backbone_very_few_params(self):
        m = CycleNet(seq_len=96, pred_len=24, enc_in=7, backbone="linear", revin=False)
        n_params = sum(p.numel() for p in m.parameters())
        # cycle: 24*7 + Linear(96→24)*7 (channel-independent) + cycle = 168 + 96*24 + 24 = ...
        # Actually Linear(96, 24) has 96*24 + 24 = 2328 params, cycle = 168, total = 2496
        assert n_params < 10_000


# ── cycle segment ─────────────────────────────────────────────────────────────


class TestCycleSegment:
    def test_wraps_correctly(self):
        m = CycleNet(seq_len=10, pred_len=5, enc_in=3, cycle_len=7)
        # Manually set cycle values
        with torch.no_grad():
            m.cycle.data = torch.arange(21, dtype=torch.float32).reshape(7, 3)
        seg = m._cycle_segment(start=5, length=4)
        # indices: 5%7=5, 6%7=6, 7%7=0, 8%7=1
        expected_rows = torch.tensor([5, 6, 0, 1])
        expected = m.cycle.data[expected_rows]
        assert torch.allclose(seg, expected)

    def test_segment_length(self):
        m = _make_model(cycle_len=24)
        seg = m._cycle_segment(0, 96)
        assert seg.shape == (96, 7)


# ── forward ───────────────────────────────────────────────────────────────────


class TestCycleNetForward:
    def test_output_shape_linear(self):
        m = _make_model(seq_len=96, pred_len=24, enc_in=7, backbone="linear")
        out = m(torch.randn(4, 96, 7))
        assert out.shape == (4, 24, 7)

    def test_output_shape_mlp(self):
        m = _make_model(seq_len=96, pred_len=24, enc_in=7, backbone="mlp")
        out = m(torch.randn(4, 96, 7))
        assert out.shape == (4, 24, 7)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        out = m(torch.randn(4, 96, 7))
        assert out.shape == (4, 24, 7)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48, 96]:
            m = CycleNet(seq_len=96, pred_len=pred_len, enc_in=7, cycle_len=24)
            out = m(torch.randn(2, 96, 7))
            assert out.shape == (2, pred_len, 7), f"Failed for pred_len={pred_len}"

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 96, 7))).all()

    def test_single_channel(self):
        m = CycleNet(seq_len=48, pred_len=12, enc_in=1, cycle_len=24)
        out = m(torch.randn(4, 48, 1))
        assert out.shape == (4, 12, 1)

    def test_gradient_flows_to_cycle(self):
        m = _make_model(revin=False)
        out = m(torch.randn(2, 96, 7))
        out.sum().backward()
        assert m.cycle.grad is not None

    def test_gradient_flows_to_backbone(self):
        m = _make_model(revin=False)
        out = m(torch.randn(2, 96, 7))
        out.sum().backward()
        assert m.backbone.weight.grad is not None

    def test_gradient_flows_to_input(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 96, 7, requires_grad=True)
        out = m(x)
        out.sum().backward()
        assert x.grad is not None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 96, 7)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_cycle_removal_reduces_variance(self):
        """After learning a cycle, residual should have lower variance than raw input
        for a perfectly periodic signal."""
        torch.manual_seed(0)
        m = CycleNet(seq_len=48, pred_len=12, enc_in=1, cycle_len=4, backbone="linear",
                     revin=False)
        # Create a perfectly periodic signal with period 4
        t = torch.arange(48, dtype=torch.float32)
        x = torch.sin(2 * torch.pi * t / 4).unsqueeze(0).unsqueeze(-1).expand(8, -1, -1)
        # Train the cycle to match the signal
        opt = torch.optim.Adam([m.cycle], lr=0.1)
        for _ in range(200):
            opt.zero_grad()
            # Loss: variance of residual
            c_past = m._cycle_segment(0, 48)
            r = x - c_past.unsqueeze(0)
            r.var().backward()
            opt.step()
        # After training, residual should have lower variance than input
        with torch.no_grad():
            c_past = m._cycle_segment(0, 48)
            residual = x - c_past.unsqueeze(0)
        assert residual.var().item() < x.var().item() * 0.1

    def test_start_token_shifts_cycle(self):
        """Different start_token values should produce different outputs
        for a model with non-zero cycle."""
        m = _make_model(revin=False)
        with torch.no_grad():
            m.cycle.data.fill_(1.0)
        x = torch.ones(2, 96, 7)
        with torch.no_grad():
            out0 = m(x, start_token=0)
            out5 = m(x, start_token=5)
        # cycle segment is always all ones (uniform cycle), so outputs should be same
        # ... unless pred segments differ. Both are all-ones so they're identical.
        # Use a non-uniform cycle to test shifting.
        with torch.no_grad():
            m.cycle.data = torch.arange(m.cycle_len, dtype=torch.float32).unsqueeze(-1).expand(-1, 7).clone()
            out_start0 = m(x, start_token=0)
            out_start3 = m(x, start_token=3)
        assert not torch.allclose(out_start0, out_start3)

    def test_batch_size_1(self):
        m = _make_model()
        out = m(torch.randn(1, 96, 7))
        assert out.shape == (1, 24, 7)


# ── registry ──────────────────────────────────────────────────────────────────


class TestCycleNetRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import CycleNet as C
        assert C is CycleNet

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.CycleNet import CycleNetForecast
        assert CycleNetForecast.model_type == "CycleNet"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("CycleNet", task)
            assert cls is not None
