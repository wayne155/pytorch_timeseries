"""Tests for DishTS — Dish-TS distribution shift normalisation."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.DishTS import DishTS, _CoSta, _TransformerLayer


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(seq_len=48, pred_len=12, enc_in=4, d_model=32, n_heads=4,
                e_layers=2, d_ff=64, dropout=0.0, dish_hidden=16):
    return DishTS(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_model=d_model, n_heads=n_heads, e_layers=e_layers,
        d_ff=d_ff, dropout=dropout, dish_hidden=dish_hidden,
    )


# ── CoSta ──────────────────────────────────────────────────────────────────────


class TestCoSta:
    def test_output_shapes(self):
        costa = _CoSta(seq_len=48, enc_in=4, hidden=16)
        x = torch.randn(2, 48, 4)
        phi_B, phi_W = costa(x)
        assert phi_B.shape == (2, 1, 4)
        assert phi_W.shape == (2, 1, 4)

    def test_phi_W_positive(self):
        costa = _CoSta(seq_len=48, enc_in=4, hidden=16)
        x = torch.randn(2, 48, 4)
        _, phi_W = costa(x)
        assert (phi_W > 0).all()

    def test_gradient_flows(self):
        costa = _CoSta(seq_len=48, enc_in=4, hidden=16)
        x = torch.randn(2, 48, 4, requires_grad=True)
        phi_B, phi_W = costa(x)
        (phi_B + phi_W).sum().backward()
        assert x.grad is not None

    def test_finite(self):
        costa = _CoSta(seq_len=48, enc_in=4, hidden=16)
        x = torch.randn(2, 48, 4)
        phi_B, phi_W = costa(x)
        assert torch.isfinite(phi_B).all()
        assert torch.isfinite(phi_W).all()


# ── TransformerLayer ───────────────────────────────────────────────────────────


class TestTransformerLayer:
    def test_output_shape(self):
        layer = _TransformerLayer(d_model=32, n_heads=4, d_ff=64, dropout=0.0)
        x = torch.randn(2, 48, 32)
        out = layer(x)
        assert out.shape == (2, 48, 32)

    def test_finite(self):
        layer = _TransformerLayer(32, 4, 64, 0.0)
        x = torch.randn(2, 48, 32)
        assert torch.isfinite(layer(x)).all()


# ── DishTS construction ───────────────────────────────────────────────────────


class TestDishTSConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_num_layers(self):
        m = _make_model(e_layers=3)
        assert len(m.layers) == 3

    def test_two_costa_networks(self):
        m = _make_model()
        assert hasattr(m, "costa_in")
        assert hasattr(m, "costa_out")

    def test_pos_embed_shape(self):
        m = _make_model(seq_len=48, d_model=32)
        assert m.pos_embed.shape == (1, 48, 32)


# ── DishTS forward ─────────────────────────────────────────────────────────────


class TestDishTSForward:
    def test_output_shape(self):
        m = _make_model(seq_len=48, pred_len=12, enc_in=4)
        out = m(torch.randn(4, 48, 4))
        assert out.shape == (4, 12, 4)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48, 96]:
            m = DishTS(seq_len=96, pred_len=pred_len, enc_in=4, d_model=32,
                       n_heads=4, e_layers=1, d_ff=64, dish_hidden=16)
            out = m(torch.randn(2, 96, 4))
            assert out.shape == (2, pred_len, 4), f"Failed pred_len={pred_len}"

    def test_single_channel(self):
        m = DishTS(seq_len=24, pred_len=8, enc_in=1, d_model=16, n_heads=4,
                   e_layers=1, d_ff=32, dish_hidden=8)
        out = m(torch.randn(2, 24, 1))
        assert out.shape == (2, 8, 1)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 48, 4))).all()

    def test_gradient_flows_to_input(self):
        m = _make_model()
        x = torch.randn(2, 48, 4, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_gradient_flows_to_costa(self):
        m = _make_model()
        x = torch.randn(2, 48, 4)
        m(x).sum().backward()
        for p in m.costa_in.parameters():
            assert p.grad is not None

    def test_batch_size_1(self):
        m = _make_model()
        out = m(torch.randn(1, 48, 4))
        assert out.shape == (1, 12, 4)

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 48, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_dish_normalisation_applied(self):
        """Input and output should differ (de-norm uses predicted statistics)."""
        torch.manual_seed(0)
        m = _make_model().eval()
        x1 = torch.randn(2, 48, 4) * 0.1
        x2 = torch.randn(2, 48, 4) * 10.0
        with torch.no_grad():
            out1 = m(x1)
            out2 = m(x2)
        assert out2.abs().mean() > out1.abs().mean()

    def test_constant_input_handled(self):
        m = _make_model().eval()
        x = torch.ones(2, 48, 4)
        with torch.no_grad():
            out = m(x)
        assert torch.isfinite(out).all()


# ── registry ───────────────────────────────────────────────────────────────────


class TestDishTSRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import DishTS as M
        assert M is DishTS

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.DishTS import DishTSForecast
        assert DishTSForecast.model_type == "DishTS"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("DishTS", task)
            assert cls is not None
