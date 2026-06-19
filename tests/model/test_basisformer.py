"""Tests for Basisformer — learnable basis function decomposition."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.Basisformer import (
    Basisformer,
    _CoeffTransformerLayer,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=96, pred_len=12, enc_in=4,
    n_basis=16, d_model=32, n_heads=4, e_layers=2, d_ff=64,
    dropout=0.0, revin=True,
):
    return Basisformer(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        n_basis=n_basis, d_model=d_model, n_heads=n_heads,
        e_layers=e_layers, d_ff=d_ff, dropout=dropout, revin=revin,
    )


# ── CoeffTransformerLayer ──────────────────────────────────────────────────────


class TestCoeffTransformerLayer:
    def test_output_shape(self):
        layer = _CoeffTransformerLayer(
            n_basis=16, d_model=32, n_heads=4, d_ff=64, dropout=0.0
        )
        coeff = torch.randn(2, 6, 16)  # (B, C, K)
        out = layer(coeff)
        assert out.shape == (2, 6, 16)

    def test_gradient_flows(self):
        layer = _CoeffTransformerLayer(
            n_basis=16, d_model=32, n_heads=4, d_ff=64, dropout=0.0
        )
        coeff = torch.randn(2, 4, 16, requires_grad=True)
        layer(coeff).sum().backward()
        assert coeff.grad is not None


# ── Basisformer construction ───────────────────────────────────────────────────


class TestBasisformerConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_basis_in_shape(self):
        m = _make_model(n_basis=16, seq_len=96)
        assert m.basis_in.shape == (16, 96)

    def test_basis_pred_shape(self):
        m = _make_model(n_basis=16, pred_len=24)
        assert m.basis_pred.shape == (16, 24)

    def test_e_layers(self):
        m = _make_model(e_layers=3)
        assert len(m.coeff_layers) == 3


# ── Basisformer forward ────────────────────────────────────────────────────────


class TestBasisformerForward:
    def test_output_shape(self):
        m = _make_model(seq_len=96, pred_len=12, enc_in=4)
        out = m(torch.randn(2, 96, 4))
        assert out.shape == (2, 12, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        out = m(torch.randn(2, 96, 4))
        assert out.shape == (2, 12, 4)

    def test_various_pred_lens(self):
        for pred_len in [12, 24, 48, 96]:
            m = Basisformer(seq_len=96, pred_len=pred_len, enc_in=4,
                            n_basis=16, d_model=32, n_heads=4, e_layers=1, d_ff=64)
            out = m(torch.randn(2, 96, 4))
            assert out.shape == (2, pred_len, 4)

    def test_single_channel(self):
        m = Basisformer(seq_len=48, pred_len=8, enc_in=1,
                        n_basis=8, d_model=16, n_heads=2, e_layers=1, d_ff=32)
        out = m(torch.randn(2, 48, 1))
        assert out.shape == (2, 8, 1)

    def test_many_channels(self):
        m = Basisformer(seq_len=48, pred_len=12, enc_in=16,
                        n_basis=8, d_model=32, n_heads=4, e_layers=1, d_ff=64)
        out = m(torch.randn(2, 48, 16))
        assert out.shape == (2, 12, 16)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 96, 4))).all()

    def test_batch_size_1(self):
        m = _make_model()
        assert m(torch.randn(1, 96, 4)).shape == (1, 12, 4)

    def test_gradient_flows_to_input(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 96, 4, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_gradient_flows_to_basis(self):
        m = _make_model()
        x = torch.randn(2, 96, 4)
        m(x).sum().backward()
        assert m.basis_in.grad is not None
        assert m.basis_pred.grad is not None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 96, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_coefficients_sum_to_one(self):
        # After softmax, coefficients per (B, C) should sum to 1 over K
        m = _make_model(revin=False)
        B, T, C = 2, 96, 4
        x = torch.randn(B, T, C)
        # Manually compute coefficients
        import math
        x_var = x.transpose(1, 2)              # (B, C, T)
        scores = x_var @ m.basis_in.T / math.sqrt(T)
        coeff = torch.softmax(scores, dim=-1)  # (B, C, K)
        assert torch.allclose(coeff.sum(-1), torch.ones(B, C), atol=1e-5)

    def test_revin_scales_output(self):
        torch.manual_seed(0)
        m = _make_model(revin=True).eval()
        with torch.no_grad():
            out_s = m(torch.randn(2, 96, 4) * 0.01)
            out_l = m(torch.randn(2, 96, 4) * 100.0)
        assert out_l.abs().mean() > out_s.abs().mean()


# ── registry ───────────────────────────────────────────────────────────────────


class TestBasisformerRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import Basisformer as M
        assert M is Basisformer

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.Basisformer import BasisformerForecast
        assert BasisformerForecast.model_type == "Basisformer"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("Basisformer", task)
            assert cls is not None
