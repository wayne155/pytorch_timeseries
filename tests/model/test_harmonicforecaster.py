"""Tests for HarmonicForecaster — differentiable spectral decomposition forecaster."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.HarmonicForecaster import HarmonicForecaster


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=64, pred_len=12, enc_in=4,
    n_harmonics=8, use_mlp=True, d_mlp=32, dropout=0.0, revin=True,
):
    return HarmonicForecaster(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        n_harmonics=n_harmonics, use_mlp=use_mlp, d_mlp=d_mlp,
        dropout=dropout, revin=revin,
    )


# ── construction ───────────────────────────────────────────────────────────────


class TestHarmonicForecasterConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_freq_query_shape(self):
        m = _make_model(seq_len=64, n_harmonics=8)
        n_freq = 64 // 2 + 1
        assert m.freq_queries.shape == (8, n_freq)

    def test_mlp_present_when_use_mlp(self):
        m = _make_model(use_mlp=True)
        assert m.mlp is not None

    def test_no_mlp_when_not_use_mlp(self):
        m = _make_model(use_mlp=False)
        assert m.mlp is None

    def test_n_harmonics_stored(self):
        m = _make_model(n_harmonics=12)
        assert m.n_harmonics == 12

    def test_n_freq_correct(self):
        m = _make_model(seq_len=64)
        assert m.n_freq == 33


# ── forward ────────────────────────────────────────────────────────────────────


class TestHarmonicForecasterForward:
    def test_output_shape(self):
        m = _make_model(seq_len=64, pred_len=12, enc_in=4)
        assert m(torch.randn(2, 64, 4)).shape == (2, 12, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        assert m(torch.randn(2, 64, 4)).shape == (2, 12, 4)

    def test_output_shape_no_mlp(self):
        m = _make_model(use_mlp=False)
        assert m(torch.randn(2, 64, 4)).shape == (2, 12, 4)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48, 96]:
            m = HarmonicForecaster(seq_len=96, pred_len=pred_len, enc_in=4,
                                   n_harmonics=8, use_mlp=True, d_mlp=32)
            assert m(torch.randn(2, 96, 4)).shape == (2, pred_len, 4)

    def test_single_channel(self):
        m = HarmonicForecaster(seq_len=32, pred_len=8, enc_in=1, n_harmonics=4)
        assert m(torch.randn(2, 32, 1)).shape == (2, 8, 1)

    def test_many_channels(self):
        m = HarmonicForecaster(seq_len=32, pred_len=8, enc_in=16, n_harmonics=4)
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

    def test_gradient_flows_to_freq_queries(self):
        m = _make_model()
        m(torch.randn(2, 64, 4)).sum().backward()
        assert m.freq_queries.grad is not None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 64, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_revin_scales_output(self):
        torch.manual_seed(0)
        m = _make_model(revin=True).eval()
        with torch.no_grad():
            out_s = m(torch.randn(2, 64, 4) * 0.01)
            out_l = m(torch.randn(2, 64, 4) * 100.0)
        assert out_l.abs().mean() > out_s.abs().mean()

    def test_pure_sinusoid_extrapolation(self):
        """On a clean sinusoid, the model should produce finite, non-trivial output."""
        m = HarmonicForecaster(seq_len=64, pred_len=16, enc_in=1,
                               n_harmonics=4, use_mlp=False, revin=False, dropout=0.0)
        t = torch.linspace(0, 4 * torch.pi, 64).unsqueeze(-1).unsqueeze(0)
        x = torch.sin(t)
        out = m(x)
        assert out.shape == (1, 16, 1)
        assert torch.isfinite(out).all()


# ── spectral properties ────────────────────────────────────────────────────────


class TestSpectralProperties:
    def test_freq_query_softmax_sums_to_one(self):
        m = _make_model()
        import torch.nn.functional as F
        w = F.softmax(m.freq_queries, dim=-1)
        assert torch.allclose(w.sum(-1), torch.ones(m.n_harmonics), atol=1e-5)

    def test_n_freq_equals_seq_len_over_2_plus_1(self):
        for seq_len in [32, 64, 96, 128]:
            m = HarmonicForecaster(seq_len=seq_len, pred_len=12, enc_in=2, n_harmonics=4)
            assert m.n_freq == seq_len // 2 + 1

    def test_t_future_starts_at_seq_len(self):
        m = _make_model(seq_len=64, pred_len=12)
        assert m.t_future[0].item() == 64.0
        assert m.t_future[-1].item() == 75.0


# ── registry ───────────────────────────────────────────────────────────────────


class TestHarmonicForecasterRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import HarmonicForecaster as M
        assert M is HarmonicForecaster

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.HarmonicForecaster import HarmonicForecasterForecast
        assert HarmonicForecasterForecast.model_type == "HarmonicForecaster"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            assert get_experiment_class("HarmonicForecaster", task) is not None
