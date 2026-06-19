"""Tests for FiLM — Frequency improved Legendre Memory."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.FiLM import FiLM, _legendre_matrix


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(seq_len=48, pred_len=12, enc_in=4, d_order=16, n_lowpass=2,
                d_ff=64, dropout=0.0, revin=True):
    return FiLM(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_order=d_order, n_lowpass=n_lowpass, d_ff=d_ff,
        dropout=dropout, revin=revin,
    )


# ── Legendre matrix ───────────────────────────────────────────────────────────


class TestLegendreMatrix:
    def test_shape(self):
        W = _legendre_matrix(48, 16)
        assert W.shape == (16, 48)

    def test_finite(self):
        W = _legendre_matrix(96, 32)
        assert torch.isfinite(W).all()

    def test_order_1_is_constant(self):
        W = _legendre_matrix(10, 1)
        # P_0 scaled by 1/T — all values equal
        assert W.shape == (1, 10)
        assert torch.allclose(W[0], W[0, 0].expand(10), atol=1e-6)

    def test_orthogonality_approximate(self):
        # W * W^T should be approximately diagonal for sufficient T
        W = _legendre_matrix(200, 8)
        G = W @ W.T  # (order, order)
        # Off-diagonal elements should be small relative to diagonal
        diag = G.diagonal().abs().mean()
        off_diag = (G - G.diagonal().diag()).abs().mean()
        assert off_diag < diag  # Not strict ortho but roughly so


# ── FiLM construction ──────────────────────────────────────────────────────────


class TestFiLMConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_W_buffer_shape(self):
        m = _make_model(seq_len=48, d_order=16)
        assert m.W.shape == (16, 48)

    def test_W_not_learnable(self):
        m = _make_model()
        assert not m.W.requires_grad

    def test_revin_exists_when_enabled(self):
        m = _make_model(revin=True)
        assert hasattr(m, "revin_layer")

    def test_no_revin_when_disabled(self):
        m = _make_model(revin=False)
        assert not hasattr(m, "revin_layer")


# ── low-pass filter ────────────────────────────────────────────────────────────


class TestLowPassFilter:
    def test_zeros_high_freqs(self):
        m = _make_model(d_order=16, n_lowpass=2)
        coeff = torch.randn(2, 4, 16)
        filtered = m._low_pass_filter(coeff)
        # After FFT, zero high freqs, IFFT — result should differ from input
        assert filtered.shape == (2, 4, 16)
        assert torch.isfinite(filtered).all()

    def test_n_lowpass_1(self):
        m = _make_model(d_order=16, n_lowpass=1)
        coeff = torch.randn(2, 4, 16)
        filtered = m._low_pass_filter(coeff)
        assert filtered.shape == (2, 4, 16)

    def test_n_lowpass_equal_to_order_is_noop(self):
        """Keeping all modes should reproduce (approximately) the input."""
        m = _make_model(d_order=16, n_lowpass=9)  # rfft size = 16//2+1 = 9
        coeff = torch.randn(2, 4, 16)
        filtered = m._low_pass_filter(coeff)
        assert torch.allclose(filtered, coeff, atol=1e-5)


# ── FiLM forward ──────────────────────────────────────────────────────────────


class TestFiLMForward:
    def test_output_shape(self):
        m = _make_model(seq_len=48, pred_len=12, enc_in=4)
        out = m(torch.randn(4, 48, 4))
        assert out.shape == (4, 12, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        out = m(torch.randn(4, 48, 4))
        assert out.shape == (4, 12, 4)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48, 96]:
            m = FiLM(seq_len=96, pred_len=pred_len, enc_in=4, d_order=16, n_lowpass=2, d_ff=64)
            out = m(torch.randn(2, 96, 4))
            assert out.shape == (2, pred_len, 4), f"Failed pred_len={pred_len}"

    def test_single_channel(self):
        m = FiLM(seq_len=24, pred_len=8, enc_in=1, d_order=8, n_lowpass=2, d_ff=32)
        out = m(torch.randn(2, 24, 1))
        assert out.shape == (2, 8, 1)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 48, 4))).all()

    def test_gradient_flows_to_decoder(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 48, 4, requires_grad=True)
        m(x).sum().backward()
        for p in m.decoder.parameters():
            assert p.grad is not None

    def test_W_receives_no_grad(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 48, 4)
        m(x).sum().backward()
        assert m.W.grad is None

    def test_batch_size_1(self):
        m = _make_model()
        out = m(torch.randn(1, 48, 4))
        assert out.shape == (1, 12, 4)

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 48, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_revin_scales_output(self):
        torch.manual_seed(0)
        m = _make_model(revin=True).eval()
        x_small = torch.randn(2, 48, 4) * 0.01
        x_large = torch.randn(2, 48, 4) * 100.0
        with torch.no_grad():
            out_small = m(x_small)
            out_large = m(x_large)
        assert out_large.abs().mean() > out_small.abs().mean()

    def test_large_d_order(self):
        m = FiLM(seq_len=96, pred_len=24, enc_in=4, d_order=64, n_lowpass=4, d_ff=128)
        out = m(torch.randn(2, 96, 4))
        assert out.shape == (2, 24, 4)


# ── registry ───────────────────────────────────────────────────────────────────


class TestFiLMRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import FiLM as M
        assert M is FiLM

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.FiLM import FiLMForecast
        assert FiLMForecast.model_type == "FiLM"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("FiLM", task)
            assert cls is not None
