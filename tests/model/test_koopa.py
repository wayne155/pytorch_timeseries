"""Tests for Koopa — Koopman operator-based non-stationary forecasting."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.Koopa import Koopa, _FourierFilter, _KoopmanPredictor


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_model(seq_len=96, pred_len=24, enc_in=7, seg_len=10, d_model=64,
                top_k=5, revin=True, dropout=0.0):
    return Koopa(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        seg_len=seg_len, d_model=d_model, top_k=top_k,
        revin=revin, dropout=dropout,
    )


# ── FourierFilter ─────────────────────────────────────────────────────────────


class TestFourierFilter:
    def test_output_shapes(self):
        ff = _FourierFilter(top_k=5)
        x = torch.randn(4, 96, 7)
        seasonal, trend = ff(x)
        assert seasonal.shape == (4, 96, 7)
        assert trend.shape == (4, 96, 7)

    def test_decomposition_is_additive(self):
        ff = _FourierFilter(top_k=5)
        x = torch.randn(2, 96, 3)
        seasonal, trend = ff(x)
        assert torch.allclose(seasonal + trend, x, atol=1e-4)

    def test_top_k_limits_seasonal_rank(self):
        """Seasonal component has at most top_k non-zero frequency bins."""
        ff = _FourierFilter(top_k=3)
        x = torch.randn(1, 32, 2)
        seasonal, _ = ff(x)
        xf = torch.fft.rfft(seasonal, dim=1)
        # Count how many frequency bins have non-trivial amplitude
        nonzero_bins = (xf.abs() > 1e-5).any(dim=(0, 2)).sum().item()
        assert nonzero_bins <= 3 + 1  # at most top_k + DC (freq=0)

    def test_seasonal_is_finite(self):
        ff = _FourierFilter(top_k=5)
        x = torch.randn(2, 96, 7)
        seasonal, trend = ff(x)
        assert torch.isfinite(seasonal).all()
        assert torch.isfinite(trend).all()


# ── KoopmanPredictor ──────────────────────────────────────────────────────────


class TestKoopmanPredictor:
    def test_output_shape_one_step(self):
        kp = _KoopmanPredictor(seg_len=10, enc_in=7, d_model=64, n_ff=128)
        x = torch.randn(4, 10, 7)
        out = kp(x, num_steps=1)
        assert out.shape == (4, 10, 7)

    def test_output_shape_multi_step(self):
        kp = _KoopmanPredictor(seg_len=10, enc_in=7, d_model=64, n_ff=128)
        x = torch.randn(4, 10, 7)
        out = kp(x, num_steps=3)
        assert out.shape == (4, 30, 7)

    def test_output_finite(self):
        kp = _KoopmanPredictor(seg_len=10, enc_in=7, d_model=64, n_ff=128)
        x = torch.randn(2, 10, 7)
        out = kp(x, num_steps=2)
        assert torch.isfinite(out).all()

    def test_gradient_flows(self):
        kp = _KoopmanPredictor(seg_len=10, enc_in=7, d_model=32, n_ff=64)
        x = torch.randn(2, 10, 7, requires_grad=True)
        out = kp(x, num_steps=2)
        out.sum().backward()
        assert x.grad is not None
        assert kp.koopman.weight.grad is not None


# ── Koopa full model ──────────────────────────────────────────────────────────


class TestKoopaConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists_when_enabled(self):
        m = _make_model(revin=True)
        assert hasattr(m, "revin_layer")

    def test_no_revin_when_disabled(self):
        m = _make_model(revin=False)
        assert not hasattr(m, "revin_layer")

    def test_num_steps_is_ceil_pred_over_seg(self):
        # pred_len=24, seg_len=10 → ceil(24/10) = 3
        m = Koopa(seq_len=96, pred_len=24, enc_in=7, seg_len=10)
        assert m.num_steps == 3

    def test_has_two_koopman_predictors(self):
        m = _make_model()
        assert hasattr(m, "seasonal_pred")
        assert hasattr(m, "trend_pred")
        assert isinstance(m.seasonal_pred, _KoopmanPredictor)
        assert isinstance(m.trend_pred, _KoopmanPredictor)

    def test_custom_n_ff(self):
        m = Koopa(seq_len=48, pred_len=12, enc_in=4, seg_len=8, d_model=32, n_ff=256)
        # encoder first linear should map seg_len*enc_in → n_ff
        assert m.seasonal_pred.encoder[0].in_features == 8 * 4
        assert m.seasonal_pred.encoder[0].out_features == 256


class TestKoopaForward:
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
            m = Koopa(seq_len=96, pred_len=pred_len, enc_in=7, seg_len=10, d_model=32)
            out = m(torch.randn(2, 96, 7))
            assert out.shape == (2, pred_len, 7), f"Failed for pred_len={pred_len}"

    def test_pred_len_not_multiple_of_seg(self):
        # pred_len=25, seg_len=10 → num_steps=3 → produces 30 → trim to 25
        m = Koopa(seq_len=96, pred_len=25, enc_in=4, seg_len=10, d_model=32)
        out = m(torch.randn(2, 96, 4))
        assert out.shape == (2, 25, 4)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 96, 7))).all()

    def test_single_channel(self):
        m = Koopa(seq_len=48, pred_len=12, enc_in=1, seg_len=8, d_model=32)
        out = m(torch.randn(4, 48, 1))
        assert out.shape == (4, 12, 1)

    def test_gradient_flows_to_input(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 96, 7, requires_grad=True)
        out = m(x)
        out.sum().backward()
        assert x.grad is not None

    def test_gradient_flows_to_koopman_operator(self):
        m = _make_model(revin=False)
        out = m(torch.randn(2, 96, 7))
        out.sum().backward()
        assert m.seasonal_pred.koopman.weight.grad is not None
        assert m.trend_pred.koopman.weight.grad is not None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 96, 7)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_revin_tracks_input_scale(self):
        torch.manual_seed(0)
        m = _make_model(revin=True).eval()
        x_small = torch.randn(2, 96, 7) * 0.01
        x_large = torch.randn(2, 96, 7) * 100.0
        with torch.no_grad():
            out_small = m(x_small)
            out_large = m(x_large)
        assert out_large.abs().mean() > out_small.abs().mean()

    def test_seg_len_equal_seq_len(self):
        # Degenerate case: single segment covers whole history
        m = Koopa(seq_len=24, pred_len=12, enc_in=3, seg_len=24, d_model=32)
        out = m(torch.randn(2, 24, 3))
        assert out.shape == (2, 12, 3)


# ── registry ──────────────────────────────────────────────────────────────────


class TestKoopaRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import Koopa as K
        assert K is Koopa

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.Koopa import KoopaForecast
        assert KoopaForecast.model_type == "Koopa"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("Koopa", task)
            assert cls is not None
