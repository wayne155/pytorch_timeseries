"""Tests for GATForecaster — Graph Attention Network with learnable edge bias."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.GATForecaster import GATForecaster, _GATLayer


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=64, pred_len=12, enc_in=4,
    d_model=32, n_heads=4, e_layers=2, d_ff=64,
    dropout=0.0, revin=True,
):
    return GATForecaster(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff,
        dropout=dropout, revin=revin,
    )


# ── GATLayer unit tests ────────────────────────────────────────────────────────


class TestGATLayer:
    def test_output_shape(self):
        gat = _GATLayer(d_model=32, n_heads=4, n_nodes=6, dropout=0.0)
        h = torch.randn(2, 6, 32)
        assert gat(h).shape == (2, 6, 32)

    def test_gradient_flows(self):
        gat = _GATLayer(d_model=16, n_heads=2, n_nodes=4, dropout=0.0)
        h = torch.randn(2, 4, 16, requires_grad=True)
        gat(h).sum().backward()
        assert h.grad is not None

    def test_edge_bias_is_parameter(self):
        gat = _GATLayer(d_model=16, n_heads=2, n_nodes=4, dropout=0.0)
        param_names = {n for n, _ in gat.named_parameters()}
        assert "edge_bias" in param_names

    def test_edge_bias_shape(self):
        gat = _GATLayer(d_model=32, n_heads=4, n_nodes=6, dropout=0.0)
        assert gat.edge_bias.shape == (4, 6, 6)

    def test_gradient_flows_to_edge_bias(self):
        gat = _GATLayer(d_model=16, n_heads=2, n_nodes=4, dropout=0.0)
        h = torch.randn(2, 4, 16)
        gat(h).sum().backward()
        assert gat.edge_bias.grad is not None


# ── GATForecaster construction ─────────────────────────────────────────────────


class TestGATForecasterConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_layer_count(self):
        assert len(_make_model(e_layers=3).layers) == 3

    def test_head_out_features(self):
        assert _make_model(pred_len=24).head.out_features == 24

    def test_embed_in_features(self):
        m = _make_model(seq_len=64)
        assert m.embed.in_features == 64


# ── GATForecaster forward ──────────────────────────────────────────────────────


class TestGATForecasterForward:
    def test_output_shape(self):
        m = _make_model(seq_len=64, pred_len=12, enc_in=4)
        assert m(torch.randn(2, 64, 4)).shape == (2, 12, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        assert m(torch.randn(2, 64, 4)).shape == (2, 12, 4)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48]:
            m = GATForecaster(seq_len=96, pred_len=pred_len, enc_in=4,
                              d_model=16, n_heads=2, e_layers=1, d_ff=32)
            assert m(torch.randn(2, 96, 4)).shape == (2, pred_len, 4)

    def test_single_channel(self):
        m = GATForecaster(seq_len=32, pred_len=8, enc_in=1,
                          d_model=16, n_heads=1, e_layers=1, d_ff=32)
        assert m(torch.randn(2, 32, 1)).shape == (2, 8, 1)

    def test_many_channels(self):
        m = GATForecaster(seq_len=32, pred_len=8, enc_in=16,
                          d_model=16, n_heads=4, e_layers=1, d_ff=32)
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

    def test_gradient_flows_to_edge_bias(self):
        m = _make_model()
        m(torch.randn(2, 64, 4)).sum().backward()
        assert m.layers[0].gat.edge_bias.grad is not None

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


# ── registry ───────────────────────────────────────────────────────────────────


class TestGATForecasterRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import GATForecaster as M
        assert M is GATForecaster

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.GATForecaster import GATForecasterForecast
        assert GATForecasterForecast.model_type == "GATForecaster"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            assert get_experiment_class("GATForecaster", task) is not None
