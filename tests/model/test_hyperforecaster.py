"""Tests for HyperForecaster — hypernetwork-generated target MLP weights."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.HyperForecaster import HyperForecaster


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=32, pred_len=8, enc_in=4,
    d_ctx=32, hidden=16, d_ctx_hidden=64, dropout=0.0, revin=True,
):
    return HyperForecaster(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_ctx=d_ctx, hidden=hidden, d_ctx_hidden=d_ctx_hidden,
        dropout=dropout, revin=revin,
    )


# ── construction ───────────────────────────────────────────────────────────────


class TestHyperForecasterConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_n_params_correct(self):
        m = _make_model(seq_len=32, pred_len=8, hidden=16)
        expected = 32 * 16 + 16 + 16 * 8 + 8  # w1 + b1 + w2 + b2
        assert m._n_params == expected

    def test_hyper_net_out_features(self):
        m = _make_model(seq_len=32, pred_len=8, hidden=16)
        expected = 32 * 16 + 16 + 16 * 8 + 8
        assert m.hyper_net.out_features == expected

    def test_ctx_encoder_is_mlp(self):
        m = _make_model()
        assert isinstance(m.ctx_encoder, nn.Sequential)

    def test_hidden_stored(self):
        m = _make_model(hidden=24)
        assert m.hidden == 24


# ── forward ────────────────────────────────────────────────────────────────────


class TestHyperForecasterForward:
    def test_output_shape(self):
        m = _make_model(seq_len=32, pred_len=8, enc_in=4)
        assert m(torch.randn(2, 32, 4)).shape == (2, 8, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        assert m(torch.randn(2, 32, 4)).shape == (2, 8, 4)

    def test_various_pred_lens(self):
        for pred_len in [8, 16, 32]:
            m = HyperForecaster(seq_len=32, pred_len=pred_len, enc_in=4,
                                d_ctx=32, hidden=16, d_ctx_hidden=64)
            assert m(torch.randn(2, 32, 4)).shape == (2, pred_len, 4)

    def test_single_channel(self):
        m = HyperForecaster(seq_len=16, pred_len=4, enc_in=1,
                            d_ctx=16, hidden=8, d_ctx_hidden=32)
        assert m(torch.randn(2, 16, 1)).shape == (2, 4, 1)

    def test_many_channels(self):
        m = HyperForecaster(seq_len=16, pred_len=4, enc_in=8,
                            d_ctx=16, hidden=8, d_ctx_hidden=32)
        assert m(torch.randn(2, 16, 8)).shape == (2, 4, 8)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 32, 4))).all()

    def test_batch_size_1(self):
        m = _make_model()
        assert m(torch.randn(1, 32, 4)).shape == (1, 8, 4)

    def test_gradient_flows_to_input(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 32, 4, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_gradient_flows_to_hyper_net(self):
        m = _make_model()
        m(torch.randn(2, 32, 4)).sum().backward()
        assert m.hyper_net.weight.grad is not None

    def test_gradient_flows_to_ctx_encoder(self):
        m = _make_model()
        m(torch.randn(2, 32, 4)).sum().backward()
        assert m.ctx_encoder[0].weight.grad is not None

    def test_different_samples_get_different_weights(self):
        """Each sample should get different target weights (non-shared)."""
        m = _make_model().eval()
        x1 = torch.randn(1, 32, 1)
        x2 = torch.randn(1, 32, 1) * 5
        ctx1 = m.ctx_encoder(x1.squeeze(-1))
        ctx2 = m.ctx_encoder(x2.squeeze(-1))
        params1 = m.hyper_net(ctx1)
        params2 = m.hyper_net(ctx2)
        assert not torch.allclose(params1, params2)

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 32, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_revin_scales_output(self):
        torch.manual_seed(0)
        m = _make_model(revin=True).eval()
        with torch.no_grad():
            out_s = m(torch.randn(2, 32, 4) * 0.01)
            out_l = m(torch.randn(2, 32, 4) * 100.0)
        assert out_l.abs().mean() > out_s.abs().mean()


# ── registry ───────────────────────────────────────────────────────────────────


class TestHyperForecasterRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import HyperForecaster as M
        assert M is HyperForecaster

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.HyperForecaster import HyperForecasterForecast
        assert HyperForecasterForecast.model_type == "HyperForecaster"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            assert get_experiment_class("HyperForecaster", task) is not None
