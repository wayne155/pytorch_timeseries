"""Tests for N-BEATS — Neural Basis Expansion model."""
import math
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.NBEATS import (
    NBEATS,
    _NBEATSBlock,
    _NBEATSStack,
    _trend_basis,
    _seasonality_basis,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=48,
    pred_len=12,
    enc_in=7,
    stack_types=None,
    num_blocks=2,
    hidden_size=64,
):
    return NBEATS(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=enc_in,
        stack_types=stack_types or ["generic"],
        num_blocks=num_blocks,
        hidden_size=hidden_size,
    )


# ── basis functions ───────────────────────────────────────────────────────────


class TestBasisFunctions:
    def test_trend_basis_shape(self):
        b = _trend_basis(24, degree=3, device=torch.device("cpu"))
        assert b.shape == (4, 24)  # degree+1 rows

    def test_trend_basis_degree_zero_is_constant(self):
        b = _trend_basis(12, degree=0, device=torch.device("cpu"))
        assert b.shape == (1, 12)
        assert torch.allclose(b, torch.ones_like(b))

    def test_seasonality_basis_shape(self):
        b = _seasonality_basis(24, num_harmonics=3, device=torch.device("cpu"))
        assert b.shape == (6, 24)  # 2*H rows

    def test_seasonality_basis_unit_norm(self):
        """Cosine/sine rows should have bounded values in [-1, 1]."""
        b = _seasonality_basis(24, num_harmonics=2, device=torch.device("cpu"))
        assert b.abs().max() <= 1.0 + 1e-6


# ── block ─────────────────────────────────────────────────────────────────────


class TestNBEATSBlock:
    @pytest.mark.parametrize("stack_type", ["generic", "trend", "seasonality"])
    def test_block_output_shapes(self, stack_type):
        block = _NBEATSBlock(
            seq_len=48, pred_len=12, enc_in=7,
            hidden_size=32, stack_type=stack_type,
        )
        x = torch.randn(4, 48, 7)
        backcast, forecast = block(x)
        assert backcast.shape == (4, 48, 7)
        assert forecast.shape == (4, 12, 7)

    def test_generic_block_gradient_flows(self):
        block = _NBEATSBlock(seq_len=24, pred_len=8, enc_in=3, hidden_size=16)
        x = torch.randn(2, 24, 3)
        backcast, forecast = block(x)
        (backcast.sum() + forecast.sum()).backward()
        assert any(p.grad is not None for p in block.parameters())

    def test_invalid_stack_type_raises(self):
        with pytest.raises(ValueError, match="stack_type"):
            _NBEATSBlock(seq_len=24, pred_len=8, enc_in=3, stack_type="wavelet")


# ── full model ────────────────────────────────────────────────────────────────


class TestNBEATSModel:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_output_shape_generic(self):
        m = _make_model(seq_len=96, pred_len=24, enc_in=7)
        out = m(torch.randn(4, 96, 7))
        assert out.shape == (4, 24, 7)

    def test_output_shape_custom_enc_in(self):
        m = _make_model(seq_len=48, pred_len=12, enc_in=3)
        out = m(torch.randn(2, 48, 3))
        assert out.shape == (2, 12, 3)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 48, 7))).all()

    def test_gradient_flows(self):
        m = _make_model()
        out = m(torch.randn(2, 48, 7))
        out.sum().backward()
        assert any(p.grad is not None for p in m.parameters())

    def test_interpretable_stack_types(self):
        m = NBEATS(
            seq_len=96, pred_len=24, enc_in=7,
            stack_types=["trend", "seasonality"],
            num_blocks=2, hidden_size=32,
        )
        out = m(torch.randn(2, 96, 7))
        assert out.shape == (2, 24, 7)

    def test_mixed_stack_types(self):
        m = NBEATS(
            seq_len=96, pred_len=24, enc_in=5,
            stack_types=["trend", "seasonality", "generic"],
            num_blocks=2, hidden_size=32,
        )
        out = m(torch.randn(3, 96, 5))
        assert out.shape == (3, 24, 5)

    def test_single_block_single_stack(self):
        m = NBEATS(
            seq_len=24, pred_len=6, enc_in=1,
            stack_types=["generic"], num_blocks=1, hidden_size=16,
        )
        out = m(torch.randn(2, 24, 1))
        assert out.shape == (2, 6, 1)

    def test_many_blocks(self):
        m = NBEATS(
            seq_len=48, pred_len=12, enc_in=7,
            stack_types=["generic", "generic"], num_blocks=5, hidden_size=32,
        )
        out = m(torch.randn(2, 48, 7))
        assert out.shape == (2, 12, 7)

    def test_default_stacks_are_generic(self):
        m = NBEATS(seq_len=48, pred_len=12, enc_in=7)
        assert m.stack_types == ["generic", "generic", "generic"]

    def test_deterministic_eval_mode(self):
        m = _make_model().eval()
        x = torch.randn(2, 48, 7)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    @pytest.mark.parametrize("pred_len", [12, 24, 48, 96])
    def test_various_pred_lens(self, pred_len):
        m = NBEATS(seq_len=96, pred_len=pred_len, enc_in=7,
                   stack_types=["generic"], hidden_size=32)
        out = m(torch.randn(2, 96, 7))
        assert out.shape == (2, pred_len, 7)


# ── registry ──────────────────────────────────────────────────────────────────


class TestNBEATSRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import NBEATS as N
        assert N is NBEATS

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.NBEATS import NBEATSForecast
        assert NBEATSForecast.model_type == "NBEATS"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("NBEATS", task)
            assert cls is not None
