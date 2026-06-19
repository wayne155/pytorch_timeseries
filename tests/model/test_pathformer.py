"""Tests for Pathformer — multi-scale adaptive path-selection transformer."""
import math

import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.Pathformer import (
    Pathformer,
    _ScaleBranch,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_model(
    seq_len=96, pred_len=12, enc_in=4,
    patch_sizes=None, d_model=32, n_heads=4, e_layers=1, d_ff=64,
    dropout=0.0, revin=True,
):
    if patch_sizes is None:
        patch_sizes = [4, 8, 16]
    return Pathformer(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        patch_sizes=patch_sizes, d_model=d_model, n_heads=n_heads,
        e_layers=e_layers, d_ff=d_ff, dropout=dropout, revin=revin,
    )


# ── ScaleBranch ────────────────────────────────────────────────────────────────


class TestScaleBranch:
    def test_output_shapes(self):
        branch = _ScaleBranch(
            seq_len=96, pred_len=12, patch_size=8,
            d_model=32, n_heads=4, e_layers=1, d_ff=64, dropout=0.0,
        )
        x_ci = torch.randn(8, 96)
        pred, pooled = branch(x_ci)
        assert pred.shape == (8, 12)
        assert pooled.shape == (8, 32)

    def test_n_patches_correct(self):
        branch = _ScaleBranch(
            seq_len=96, pred_len=12, patch_size=8,
            d_model=32, n_heads=4, e_layers=1, d_ff=64, dropout=0.0,
        )
        assert branch.n_patches == math.ceil(96 / 8)

    def test_short_seq_padded(self):
        # seq_len not divisible by patch_size
        branch = _ScaleBranch(
            seq_len=10, pred_len=4, patch_size=8,
            d_model=16, n_heads=2, e_layers=1, d_ff=32, dropout=0.0,
        )
        x_ci = torch.randn(4, 10)
        pred, pooled = branch(x_ci)
        assert pred.shape == (4, 4)

    def test_gradient_flows(self):
        branch = _ScaleBranch(
            seq_len=48, pred_len=8, patch_size=8,
            d_model=16, n_heads=2, e_layers=1, d_ff=32, dropout=0.0,
        )
        x = torch.randn(4, 48, requires_grad=True)
        pred, _ = branch(x)
        pred.sum().backward()
        assert x.grad is not None


# ── Pathformer construction ────────────────────────────────────────────────────


class TestPathformerConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_revin_exists_when_enabled(self):
        assert hasattr(_make_model(revin=True), "revin_layer")

    def test_no_revin_when_disabled(self):
        assert not hasattr(_make_model(revin=False), "revin_layer")

    def test_num_branches(self):
        m = _make_model(patch_sizes=[4, 8, 16])
        assert len(m.branches) == 3

    def test_single_scale(self):
        m = _make_model(patch_sizes=[8])
        assert len(m.branches) == 1

    def test_router_in_features(self):
        m = _make_model(patch_sizes=[4, 8, 16], d_model=32)
        router_fc = m.router[0]
        assert router_fc.in_features == 3 * 32

    def test_router_out_features(self):
        m = _make_model(patch_sizes=[4, 8, 16])
        router_last = m.router[-1]
        assert router_last.out_features == 3


# ── Pathformer forward ─────────────────────────────────────────────────────────


class TestPathformerForward:
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
            m = Pathformer(seq_len=96, pred_len=pred_len, enc_in=4,
                           patch_sizes=[4, 8, 16], d_model=32, n_heads=4,
                           e_layers=1, d_ff=64)
            out = m(torch.randn(2, 96, 4))
            assert out.shape == (2, pred_len, 4), f"Failed pred_len={pred_len}"

    def test_single_channel(self):
        m = Pathformer(seq_len=48, pred_len=8, enc_in=1,
                       patch_sizes=[4, 8], d_model=16, n_heads=2,
                       e_layers=1, d_ff=32)
        out = m(torch.randn(2, 48, 1))
        assert out.shape == (2, 8, 1)

    def test_many_channels(self):
        m = Pathformer(seq_len=48, pred_len=12, enc_in=16,
                       patch_sizes=[4, 8], d_model=32, n_heads=4,
                       e_layers=1, d_ff=64)
        out = m(torch.randn(2, 48, 16))
        assert out.shape == (2, 12, 16)

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 96, 4))).all()

    def test_batch_size_1(self):
        m = _make_model()
        out = m(torch.randn(1, 96, 4))
        assert out.shape == (1, 12, 4)

    def test_gradient_flows_to_input(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 96, 4, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_gradient_flows_to_router(self):
        m = _make_model()
        x = torch.randn(2, 96, 4)
        m(x).sum().backward()
        assert m.router[0].weight.grad is not None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 96, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_router_weights_sum_to_one(self):
        m = _make_model(patch_sizes=[4, 8, 16], d_model=32, revin=False)
        # Access router weights by hooking into branches
        B, T, C = 2, 96, 4
        x = torch.randn(B, T, C)
        x_ci = x.transpose(1, 2).reshape(B * C, T)
        pooled_list = []
        for branch in m.branches:
            _, pooled_k = branch(x_ci)
            pooled_list.append(pooled_k)
        pooled_cat = torch.cat(pooled_list, dim=-1)
        weights = torch.softmax(m.router(pooled_cat), dim=-1)
        assert torch.allclose(weights.sum(-1), torch.ones(B * C), atol=1e-5)

    def test_short_seq_with_large_patch(self):
        # patch size larger than seq_len: should still work with padding
        m = Pathformer(seq_len=8, pred_len=4, enc_in=2,
                       patch_sizes=[16], d_model=16, n_heads=2,
                       e_layers=1, d_ff=32)
        out = m(torch.randn(2, 8, 2))
        assert out.shape == (2, 4, 2)

    def test_revin_scales_output(self):
        torch.manual_seed(0)
        m = _make_model(revin=True).eval()
        with torch.no_grad():
            out_s = m(torch.randn(2, 96, 4) * 0.01)
            out_l = m(torch.randn(2, 96, 4) * 100.0)
        assert out_l.abs().mean() > out_s.abs().mean()


# ── registry ───────────────────────────────────────────────────────────────────


class TestPathformerRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import Pathformer as M
        assert M is Pathformer

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.Pathformer import PathformerForecast
        assert PathformerForecast.model_type == "Pathformer"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("Pathformer", task)
            assert cls is not None
