"""Tests for HDMixer — Hierarchical Dependency MLP Mixer."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.HDMixer import HDMixer, _ScaleMixer


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=64, pred_len=12, enc_in=4,
    patch_sizes=None, d_model=32, dropout=0.0, revin=True,
):
    return HDMixer(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        patch_sizes=patch_sizes or [4, 8, 16],
        d_model=d_model, dropout=dropout, revin=revin,
    )


# ── ScaleMixer unit tests ──────────────────────────────────────────────────────


class TestScaleMixer:
    def test_output_shape(self):
        sm = _ScaleMixer(seq_len=64, patch_size=8, d_model=32, dropout=0.0)
        out = sm(torch.randn(6, 64))
        assert out.shape == (6, 32)

    def test_output_shape_non_divisible(self):
        sm = _ScaleMixer(seq_len=63, patch_size=8, d_model=32, dropout=0.0)
        out = sm(torch.randn(4, 63))
        assert out.shape == (4, 32)

    def test_gradient_flows(self):
        sm = _ScaleMixer(seq_len=32, patch_size=4, d_model=16, dropout=0.0)
        x = torch.randn(4, 32, requires_grad=True)
        sm(x).sum().backward()
        assert x.grad is not None

    def test_padding_computed_correctly(self):
        sm = _ScaleMixer(seq_len=30, patch_size=8, d_model=16, dropout=0.0)
        # ceil(30/8)*8 - 30 = 4*8 - 30 = 2
        assert sm.pad_len == 2
        assert sm.n_patches == 4


# ── HDMixer construction ───────────────────────────────────────────────────────


class TestHDMixerConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_n_scales(self):
        m = _make_model(patch_sizes=[4, 8, 16])
        assert len(m.mixers) == 3

    def test_single_scale(self):
        m = _make_model(patch_sizes=[8])
        assert len(m.mixers) == 1

    def test_head_out_features(self):
        assert _make_model(pred_len=24).head.out_features == 24


# ── HDMixer forward ────────────────────────────────────────────────────────────


class TestHDMixerForward:
    def test_output_shape(self):
        m = _make_model(seq_len=64, pred_len=12, enc_in=4)
        assert m(torch.randn(2, 64, 4)).shape == (2, 12, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        assert m(torch.randn(2, 64, 4)).shape == (2, 12, 4)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48, 96]:
            m = HDMixer(seq_len=96, pred_len=pred_len, enc_in=4,
                        patch_sizes=[8, 16], d_model=32)
            assert m(torch.randn(2, 96, 4)).shape == (2, pred_len, 4)

    def test_single_channel(self):
        m = HDMixer(seq_len=32, pred_len=8, enc_in=1, patch_sizes=[4, 8])
        assert m(torch.randn(2, 32, 1)).shape == (2, 8, 1)

    def test_many_channels(self):
        m = HDMixer(seq_len=32, pred_len=8, enc_in=16, patch_sizes=[4, 8])
        assert m(torch.randn(2, 32, 16)).shape == (2, 8, 16)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 64, 4))).all()

    def test_batch_size_1(self):
        m = _make_model()
        assert m(torch.randn(1, 64, 4)).shape == (1, 12, 4)

    def test_gradient_flows_to_input(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 64, 4, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 64, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_non_divisible_seq_len(self):
        m = HDMixer(seq_len=63, pred_len=12, enc_in=4, patch_sizes=[4, 8, 16])
        assert m(torch.randn(2, 63, 4)).shape == (2, 12, 4)

    def test_revin_scales_output(self):
        torch.manual_seed(0)
        m = _make_model(revin=True).eval()
        with torch.no_grad():
            out_s = m(torch.randn(2, 64, 4) * 0.01)
            out_l = m(torch.randn(2, 64, 4) * 100.0)
        assert out_l.abs().mean() > out_s.abs().mean()


# ── registry ───────────────────────────────────────────────────────────────────


class TestHDMixerRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import HDMixer as M
        assert M is HDMixer

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.HDMixer import HDMixerForecast
        assert HDMixerForecast.model_type == "HDMixer"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            assert get_experiment_class("HDMixer", task) is not None
