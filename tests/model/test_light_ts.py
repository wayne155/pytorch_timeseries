"""Tests for LightTS — dual-sampling interval-enhanced MLP forecaster."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.LightTS import LightTS, _IEBlock


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_model(seq_len=96, pred_len=24, enc_in=7, chunk_size=8, d_model=64,
                revin=True, dropout=0.0):
    return LightTS(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        chunk_size=chunk_size, d_model=d_model, revin=revin, dropout=dropout,
    )


# ── IEBlock ───────────────────────────────────────────────────────────────────


class TestIEBlock:
    def test_output_shape(self):
        block = _IEBlock(input_dim=12, hid_dim=64, out_dim=32, num_node=8)
        x = torch.randn(4, 8, 12)    # (B*N, L, seg_len)
        out = block(x)
        assert out.shape == (4, 8, 32)

    def test_output_finite(self):
        block = _IEBlock(input_dim=12, hid_dim=64, out_dim=32, num_node=8)
        x = torch.randn(4, 8, 12)
        assert torch.isfinite(block(x)).all()

    def test_gradient_flows(self):
        block = _IEBlock(input_dim=12, hid_dim=64, out_dim=32, num_node=8)
        x = torch.randn(2, 8, 12, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None


# ── construction ──────────────────────────────────────────────────────────────


class TestLightTSConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_default_chunk_size(self):
        m = LightTS(seq_len=96, pred_len=24, enc_in=7)
        assert m.L == 8   # min(seq_len, 8)

    def test_custom_chunk_size(self):
        m = LightTS(seq_len=96, pred_len=24, enc_in=7, chunk_size=4)
        assert m.L == 4

    def test_seg_len_is_seq_over_L(self):
        m = LightTS(seq_len=96, pred_len=24, enc_in=7, chunk_size=8)
        assert m.seg_len == 12   # 96 / 8

    def test_revin_exists_when_enabled(self):
        m = _make_model(revin=True)
        assert hasattr(m, "revin_layer")

    def test_no_revin_when_disabled(self):
        m = _make_model(revin=False)
        assert not hasattr(m, "revin_layer")

    def test_has_cont_and_intv_blocks(self):
        m = _make_model()
        assert hasattr(m, "cont_block")
        assert hasattr(m, "intv_block")
        assert isinstance(m.cont_block, _IEBlock)
        assert isinstance(m.intv_block, _IEBlock)

    def test_projection_maps_to_pred_len(self):
        m = _make_model(pred_len=48)
        assert m.projection.out_features == 48

    def test_small_param_count(self):
        """LightTS should have far fewer params than attention models."""
        m = LightTS(seq_len=96, pred_len=24, enc_in=7, chunk_size=8, d_model=64, revin=False)
        n_params = sum(p.numel() for p in m.parameters())
        # Should be in the low thousands, not millions
        assert n_params < 200_000


# ── forward ───────────────────────────────────────────────────────────────────


class TestLightTSForward:
    def test_output_shape(self):
        m = _make_model(seq_len=96, pred_len=24, enc_in=7)
        out = m(torch.randn(4, 96, 7))
        assert out.shape == (4, 24, 7)

    def test_output_shape_no_revin(self):
        m = _make_model(seq_len=96, pred_len=24, enc_in=7, revin=False)
        out = m(torch.randn(4, 96, 7))
        assert out.shape == (4, 24, 7)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48, 96]:
            m = LightTS(seq_len=96, pred_len=pred_len, enc_in=7, chunk_size=8, d_model=32)
            out = m(torch.randn(2, 96, 7))
            assert out.shape == (2, pred_len, 7), f"Failed for pred_len={pred_len}"

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 96, 7))).all()

    def test_single_channel(self):
        m = LightTS(seq_len=48, pred_len=12, enc_in=1, chunk_size=4, d_model=32)
        out = m(torch.randn(4, 48, 1))
        assert out.shape == (4, 12, 1)

    def test_many_channels(self):
        m = LightTS(seq_len=48, pred_len=12, enc_in=100, chunk_size=4, d_model=32)
        out = m(torch.randn(2, 48, 100))
        assert out.shape == (2, 12, 100)

    def test_gradient_flows_to_input(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 96, 7, requires_grad=True)
        out = m(x)
        out.sum().backward()
        assert x.grad is not None

    def test_gradient_flows_to_projection(self):
        m = _make_model(revin=False)
        out = m(torch.randn(2, 96, 7))
        out.sum().backward()
        assert m.projection.weight.grad is not None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 96, 7)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_batch_size_1(self):
        m = _make_model()
        out = m(torch.randn(1, 96, 7))
        assert out.shape == (1, 24, 7)

    def test_chunk_size_1(self):
        # Degenerate: L=1, one chunk = whole sequence
        m = LightTS(seq_len=16, pred_len=8, enc_in=3, chunk_size=1, d_model=32)
        out = m(torch.randn(2, 16, 3))
        assert out.shape == (2, 8, 3)

    def test_revin_tracks_input_scale(self):
        torch.manual_seed(0)
        m = _make_model(revin=True).eval()
        x_small = torch.randn(2, 96, 7) * 0.01
        x_large = torch.randn(2, 96, 7) * 100.0
        with torch.no_grad():
            out_small = m(x_small)
            out_large = m(x_large)
        assert out_large.abs().mean() > out_small.abs().mean()

    def test_continuous_and_interval_views_differ(self):
        """Continuous and interval blocks see differently ordered inputs."""
        m = _make_model(revin=False)
        # Monkey-patch to check block inputs differ for non-constant input
        cont_inputs, intv_inputs = [], []
        orig_cont = m.cont_block.forward
        orig_intv = m.intv_block.forward

        def hook_cont(x):
            cont_inputs.append(x.detach())
            return orig_cont(x)

        def hook_intv(x):
            intv_inputs.append(x.detach())
            return orig_intv(x)

        m.cont_block.forward = hook_cont
        m.intv_block.forward = hook_intv
        m(torch.randn(2, 96, 7))
        # Both blocks see same shape
        assert cont_inputs[0].shape == intv_inputs[0].shape
        # But different content (except with constant input)
        assert not torch.allclose(cont_inputs[0], intv_inputs[0])


# ── registry ──────────────────────────────────────────────────────────────────


class TestLightTSRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import LightTS as L
        assert L is LightTS

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.LightTS import LightTSForecast
        assert LightTSForecast.model_type == "LightTS"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("LightTS", task)
            assert cls is not None
