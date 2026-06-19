"""Tests for KANForecaster — Chebyshev KAN for time-series forecasting."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from torch_timeseries.model.KANForecaster import KANForecaster, _ChebyKANLayer


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=64, pred_len=12, enc_in=4,
    hidden=32, e_layers=2, degree=3, dropout=0.0, revin=True,
):
    return KANForecaster(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        hidden=hidden, e_layers=e_layers, degree=degree,
        dropout=dropout, revin=revin,
    )


# ── ChebyKANLayer unit tests ───────────────────────────────────────────────────


class TestChebyKANLayer:
    def test_output_shape(self):
        layer = _ChebyKANLayer(16, 8, degree=3)
        assert layer(torch.randn(4, 16)).shape == (4, 8)

    def test_gradient_flows(self):
        layer = _ChebyKANLayer(8, 4, degree=3)
        x = torch.randn(3, 8, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None
        assert layer.coeffs.grad is not None

    def test_coeffs_shape(self):
        layer = _ChebyKANLayer(6, 4, degree=5)
        assert layer.coeffs.shape == (4, 6, 6)  # (out, in, degree+1)

    def test_chebyshev_basis_shape(self):
        layer = _ChebyKANLayer(8, 4, degree=4)
        x = torch.randn(3, 8)
        basis = layer._chebyshev_basis(x.clamp(-1, 1))
        assert basis.shape == (3, 8, 5)

    def test_chebyshev_t0_is_ones(self):
        layer = _ChebyKANLayer(4, 2, degree=3)
        x = torch.randn(5, 4).clamp(-1, 1)
        basis = layer._chebyshev_basis(x)
        assert torch.allclose(basis[:, :, 0], torch.ones(5, 4))

    def test_chebyshev_t1_equals_x(self):
        layer = _ChebyKANLayer(4, 2, degree=3)
        x = torch.randn(5, 4).clamp(-1, 1)
        basis = layer._chebyshev_basis(x)
        assert torch.allclose(basis[:, :, 1], x)

    def test_normalisation_keeps_in_range(self):
        layer = _ChebyKANLayer(8, 4, degree=3)
        x = torch.randn(4, 8) * 100  # large values
        out = layer(x)
        assert torch.isfinite(out).all()

    def test_different_degrees(self):
        for deg in [1, 3, 5, 7]:
            layer = _ChebyKANLayer(8, 4, degree=deg)
            assert layer(torch.randn(2, 8)).shape == (2, 4)


# ── KANForecaster construction ─────────────────────────────────────────────────


class TestKANForecasterConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_layer_count(self):
        # e_layers KAN layers + 1 head = e_layers+1 total ChebyKAN
        m = _make_model(e_layers=2, dropout=0.0)
        kan_layers = [l for l in m.kan_net if isinstance(l, _ChebyKANLayer)]
        assert len(kan_layers) == 3  # 2 hidden + 1 head

    def test_dropout_layers_present(self):
        m = _make_model(e_layers=2, dropout=0.1)
        drop_layers = [l for l in m.kan_net if isinstance(l, nn.Dropout)]
        assert len(drop_layers) == 2


# ── KANForecaster forward ──────────────────────────────────────────────────────


class TestKANForecasterForward:
    def test_output_shape(self):
        m = _make_model(seq_len=64, pred_len=12, enc_in=4)
        assert m(torch.randn(2, 64, 4)).shape == (2, 12, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        assert m(torch.randn(2, 64, 4)).shape == (2, 12, 4)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48, 96]:
            m = KANForecaster(seq_len=96, pred_len=pred_len, enc_in=4,
                              hidden=32, e_layers=1, degree=3)
            assert m(torch.randn(2, 96, 4)).shape == (2, pred_len, 4)

    def test_single_channel(self):
        m = KANForecaster(seq_len=32, pred_len=8, enc_in=1, hidden=16, e_layers=1, degree=3)
        assert m(torch.randn(2, 32, 1)).shape == (2, 8, 1)

    def test_many_channels(self):
        m = KANForecaster(seq_len=32, pred_len=8, enc_in=16, hidden=16, e_layers=1, degree=3)
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

    def test_gradient_flows_to_coeffs(self):
        m = _make_model()
        m(torch.randn(2, 64, 4)).sum().backward()
        first_kan = next(l for l in m.kan_net if isinstance(l, _ChebyKANLayer))
        assert first_kan.coeffs.grad is not None

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

    def test_large_input_stable(self):
        m = _make_model(revin=False)
        assert torch.isfinite(m(torch.randn(2, 64, 4) * 1000)).all()


# ── registry ───────────────────────────────────────────────────────────────────


class TestKANForecasterRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import KANForecaster as M
        assert M is KANForecaster

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.KANForecaster import KANForecasterForecast
        assert KANForecasterForecast.model_type == "KANForecaster"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            assert get_experiment_class("KANForecaster", task) is not None
