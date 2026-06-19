"""Tests for GCNForecaster — adaptive graph convolutional network forecaster."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.GCNForecaster import (
    GCNForecaster,
    _AdaptiveAdjacency,
    _GraphConvLayer,
    _TemporalBlock,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=48, pred_len=12, enc_in=4,
    d_model=32, e_layers=2, d_emb=8, k_hops=2,
    kernel_size=3, dropout=0.0, revin=True,
):
    return GCNForecaster(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_model=d_model, e_layers=e_layers, d_emb=d_emb, k_hops=k_hops,
        kernel_size=kernel_size, dropout=dropout, revin=revin,
    )


# ── AdaptiveAdjacency ──────────────────────────────────────────────────────────


class TestAdaptiveAdjacency:
    def test_output_shape(self):
        adj = _AdaptiveAdjacency(n_nodes=4, d_emb=8)
        A = adj()
        assert A.shape == (4, 4)

    def test_row_sums_to_one(self):
        adj = _AdaptiveAdjacency(n_nodes=4, d_emb=8)
        A = adj()
        assert torch.allclose(A.sum(-1), torch.ones(4), atol=1e-5)

    def test_non_negative(self):
        adj = _AdaptiveAdjacency(n_nodes=6, d_emb=4)
        A = adj()
        assert (A >= 0).all()

    def test_gradient_flows(self):
        adj = _AdaptiveAdjacency(n_nodes=4, d_emb=8)
        A = adj()
        A.sum().backward()
        assert adj.E1.grad is not None


# ── GraphConvLayer ─────────────────────────────────────────────────────────────


class TestGraphConvLayer:
    def test_output_shape(self):
        gcn = _GraphConvLayer(d_in=32, d_out=32, k_hops=2, dropout=0.0)
        h = torch.randn(2, 4, 32)
        A = torch.softmax(torch.randn(4, 4), dim=-1)
        out = gcn(h, A)
        assert out.shape == (2, 4, 32)

    def test_gradient_flows(self):
        gcn = _GraphConvLayer(d_in=16, d_out=16, k_hops=1, dropout=0.0)
        h = torch.randn(2, 3, 16, requires_grad=True)
        A = torch.softmax(torch.randn(3, 3), dim=-1)
        gcn(h, A).sum().backward()
        assert h.grad is not None

    def test_zero_hop(self):
        gcn = _GraphConvLayer(d_in=16, d_out=16, k_hops=0, dropout=0.0)
        h = torch.randn(2, 4, 16)
        A = torch.eye(4)
        out = gcn(h, A)
        assert out.shape == (2, 4, 16)


# ── TemporalBlock ──────────────────────────────────────────────────────────────


class TestTemporalBlock:
    def test_output_shape_same_T(self):
        block = _TemporalBlock(d_model=32, kernel_size=3, dilation=1, dropout=0.0)
        x = torch.randn(8, 32, 48)
        out = block(x)
        assert out.shape == (8, 32, 48)

    def test_gradient_flows(self):
        block = _TemporalBlock(d_model=16, kernel_size=3, dilation=2, dropout=0.0)
        x = torch.randn(4, 16, 24, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None


# ── GCNForecaster construction ─────────────────────────────────────────────────


class TestGCNForecasterConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_layer_counts(self):
        m = _make_model(e_layers=3)
        assert len(m.temp_blocks) == 3
        assert len(m.gcn_layers) == 3

    def test_output_proj_shape(self):
        m = _make_model(seq_len=48, pred_len=24, d_model=32)
        assert m.output_proj.in_features == 32 * 48
        assert m.output_proj.out_features == 24


# ── GCNForecaster forward ──────────────────────────────────────────────────────


class TestGCNForecasterForward:
    def test_output_shape(self):
        m = _make_model(seq_len=48, pred_len=12, enc_in=4)
        out = m(torch.randn(2, 48, 4))
        assert out.shape == (2, 12, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        out = m(torch.randn(2, 48, 4))
        assert out.shape == (2, 12, 4)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48]:
            m = GCNForecaster(seq_len=48, pred_len=pred_len, enc_in=4,
                              d_model=16, e_layers=1, d_emb=4, k_hops=1)
            out = m(torch.randn(2, 48, 4))
            assert out.shape == (2, pred_len, 4)

    def test_single_channel(self):
        m = GCNForecaster(seq_len=24, pred_len=8, enc_in=1,
                          d_model=16, e_layers=1, d_emb=4, k_hops=1)
        out = m(torch.randn(2, 24, 1))
        assert out.shape == (2, 8, 1)

    def test_many_channels(self):
        m = GCNForecaster(seq_len=24, pred_len=8, enc_in=16,
                          d_model=32, e_layers=1, d_emb=8, k_hops=1)
        out = m(torch.randn(2, 24, 16))
        assert out.shape == (2, 8, 16)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 48, 4))).all()

    def test_batch_size_1(self):
        m = _make_model()
        assert m(torch.randn(1, 48, 4)).shape == (1, 12, 4)

    def test_gradient_flows_to_input(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 48, 4, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_gradient_flows_to_adjacency(self):
        m = _make_model()
        x = torch.randn(2, 48, 4)
        m(x).sum().backward()
        assert m.adj.E1.grad is not None
        assert m.adj.E2.grad is not None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 48, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_revin_scales_output(self):
        torch.manual_seed(0)
        m = _make_model(revin=True).eval()
        with torch.no_grad():
            out_s = m(torch.randn(2, 48, 4) * 0.01)
            out_l = m(torch.randn(2, 48, 4) * 100.0)
        assert out_l.abs().mean() > out_s.abs().mean()

    def test_adjacency_is_row_normalised(self):
        m = _make_model(enc_in=4)
        A = m.adj()
        assert torch.allclose(A.sum(-1), torch.ones(4), atol=1e-5)


# ── registry ───────────────────────────────────────────────────────────────────


class TestGCNForecasterRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import GCNForecaster as M
        assert M is GCNForecaster

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.GCNForecaster import GCNForecasterForecast
        assert GCNForecasterForecast.model_type == "GCNForecaster"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("GCNForecaster", task)
            assert cls is not None
