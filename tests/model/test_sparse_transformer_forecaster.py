"""Tests for SparseTransformerForecaster — local+strided sparse attention."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.SparseTransformerForecaster import (
    SparseTransformerForecaster,
    _build_sparse_mask,
)


# ── helpers ───────────────────────────���──────────────────────────��─────────────


def _make_model(
    seq_len=32, pred_len=8, enc_in=4,
    patch_size=8, d_model=32, n_heads=4, e_layers=2, d_ff=64,
    local_window=2, stride=2, dropout=0.1, revin=True,
):
    return SparseTransformerForecaster(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        patch_size=patch_size, d_model=d_model, n_heads=n_heads,
        e_layers=e_layers, d_ff=d_ff,
        local_window=local_window, stride=stride,
        dropout=dropout, revin=revin,
    )


# ── sparse mask ──────────────────��────────────────────────────���────────────────


class TestSparseMask:
    def test_local_window_is_attended(self):
        mask = _build_sparse_mask(8, local_window=1, stride=100)
        # Token 3 should attend to 2 and 4
        assert mask[3, 2] == 0.0
        assert mask[3, 4] == 0.0

    def test_far_non_stride_is_masked(self):
        # stride=100 means no token besides 0 is a stride token
        mask = _build_sparse_mask(8, local_window=1, stride=100)
        # Token 1 (not a stride token) should not attend to token 7 (outside window)
        assert mask[1, 7] == float("-inf")

    def test_stride_token_attends_all(self):
        mask = _build_sparse_mask(8, local_window=1, stride=4)
        # Token 0 (stride token) should attend to all
        assert (mask[0] == 0.0).all()

    def test_all_attend_stride_token(self):
        mask = _build_sparse_mask(8, local_window=1, stride=4)
        # All tokens should attend to token 0 (stride=4, index 0 is multiple)
        assert (mask[:, 0] == 0.0).all()

    def test_diagonal_always_attended(self):
        mask = _build_sparse_mask(6, local_window=0, stride=100)
        for i in range(6):
            assert mask[i, i] == 0.0


# ── construction ─────────────────────────��─────────────────────────────��───────


class TestSparseTransformerForecasterConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_sparse_mask_is_buffer(self):
        m = _make_model()
        buffer_names = {n for n, _ in m.named_buffers()}
        assert any("sparse_mask" in n for n in buffer_names)

    def test_sparse_mask_not_parameter(self):
        m = _make_model()
        param_names = {n for n, _ in m.named_parameters()}
        assert not any("sparse_mask" in n for n in param_names)

    def test_patch_embed_input(self):
        m = _make_model(patch_size=8)
        assert m.patch_embed.in_features == 8

    def test_pos_embed_shape(self):
        m = _make_model(seq_len=32, patch_size=8, d_model=32)
        # n_patches = 32/8 = 4
        assert m.pos_embed.shape == (1, 4, 32)


# ── forward ─────────────────────��─────────────────────────────���────────────────


class TestSparseTransformerForecasterForward:
    def test_output_shape(self):
        m = _make_model(seq_len=32, pred_len=8, enc_in=4)
        assert m(torch.randn(2, 32, 4)).shape == (2, 8, 4)

    def test_output_shape_no_revin(self):
        m = _make_model(revin=False)
        assert m(torch.randn(2, 32, 4)).shape == (2, 8, 4)

    def test_single_channel(self):
        m = SparseTransformerForecaster(seq_len=16, pred_len=4, enc_in=1, patch_size=4, d_model=16, n_heads=4)
        assert m(torch.randn(2, 16, 1)).shape == (2, 4, 1)

    def test_many_channels(self):
        m = SparseTransformerForecaster(seq_len=16, pred_len=4, enc_in=8, patch_size=4, d_model=16, n_heads=4)
        assert m(torch.randn(2, 16, 8)).shape == (2, 4, 8)

    def test_seq_len_non_divisible_by_patch(self):
        # seq_len=30, patch_size=8 → pad 2 → 32/8=4 patches
        m = SparseTransformerForecaster(seq_len=30, pred_len=6, enc_in=4, patch_size=8, d_model=16, n_heads=4)
        assert m(torch.randn(2, 30, 4)).shape == (2, 6, 4)

    def test_various_pred_lens(self):
        for pred_len in [8, 24, 96]:
            m = SparseTransformerForecaster(seq_len=32, pred_len=pred_len, enc_in=4,
                                            patch_size=8, d_model=16, n_heads=4)
            assert m(torch.randn(2, 32, 4)).shape == (2, pred_len, 4)

    def test_batch_size_1(self):
        m = _make_model()
        assert m(torch.randn(1, 32, 4)).shape == (1, 8, 4)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 32, 4))).all()

    def test_gradient_flows(self):
        m = _make_model()
        m(torch.randn(2, 32, 4)).sum().backward()
        assert m.patch_embed.weight.grad is not None
        assert m.head.weight.grad is not None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 32, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))


# ── registry ──────────────────────────────────────────────��────────────────────


class TestSparseTransformerForecasterRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import SparseTransformerForecaster as M
        assert M is SparseTransformerForecaster

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.SparseTransformerForecaster import SparseTransformerForecasterForecast
        assert SparseTransformerForecasterForecast.model_type == "SparseTransformerForecaster"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            assert get_experiment_class("SparseTransformerForecaster", task) is not None
