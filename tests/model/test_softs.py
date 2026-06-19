"""Tests for SOFTS — star-topology Series-Core Fusion forecasting model."""
import torch
import torch.nn as nn
import pytest

from torch_timeseries.model.SOFTS import SOFTS, _SOFTSBlock


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_model(seq_len=96, pred_len=24, enc_in=7, d_model=64, d_core=None,
                e_layers=2, dropout=0.0, revin=True):
    return SOFTS(
        seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
        d_model=d_model, d_core=d_core, e_layers=e_layers,
        dropout=dropout, revin=revin,
    )


# ── construction ──────────────────────────────────────────────────────────────


class TestSOFTSConstruction:
    def test_is_nn_module(self):
        assert isinstance(_make_model(), nn.Module)

    def test_default_d_core_equals_d_model(self):
        m = SOFTS(seq_len=48, pred_len=12, enc_in=4, d_model=32)
        # each block's core_mlp first linear should map d_model → d_model
        block = m.blocks[0]
        assert block.core_mlp[1].in_features == 32
        assert block.core_mlp[1].out_features == 32

    def test_custom_d_core(self):
        m = SOFTS(seq_len=48, pred_len=12, enc_in=4, d_model=64, d_core=16)
        block = m.blocks[0]
        assert block.core_mlp[1].out_features == 16

    def test_num_blocks_matches_e_layers(self):
        m = _make_model(e_layers=3)
        assert len(m.blocks) == 3

    def test_revin_exists_when_enabled(self):
        m = _make_model(revin=True)
        assert hasattr(m, "revin_layer")

    def test_no_revin_when_disabled(self):
        m = _make_model(revin=False)
        assert not hasattr(m, "revin_layer")

    def test_input_proj_maps_seq_to_d_model(self):
        m = _make_model(seq_len=96, d_model=64)
        assert m.input_proj.in_features == 96
        assert m.input_proj.out_features == 64

    def test_output_proj_maps_d_model_to_pred_len(self):
        m = _make_model(d_model=64, pred_len=48)
        assert m.output_proj.in_features == 64
        assert m.output_proj.out_features == 48


# ── SOFTSBlock ────────────────────────────────────────────────────────────────


class TestSOFTSBlock:
    def test_output_shape_matches_input(self):
        B, N, D = 4, 7, 32
        block = _SOFTSBlock(d_model=D, d_core=D)
        x = torch.randn(B, N, D)
        out = block(x)
        assert out.shape == (B, N, D)

    def test_core_is_shared_across_variates(self):
        """Core is the mean over N variates so it broadcasts back uniformly."""
        B, N, D = 1, 5, 16
        block = _SOFTSBlock(d_model=D, d_core=D).eval()
        x = torch.ones(B, N, D)
        # All variates identical → mean = same → fusion identical for all
        out = block(x)
        # All N output embeddings should be the same (all ones input + same core)
        assert torch.allclose(out[:, 0, :], out[:, 1, :], atol=1e-5)

    def test_gradient_flows_through_core(self):
        B, N, D = 2, 6, 32
        block = _SOFTSBlock(d_model=D, d_core=D)
        x = torch.randn(B, N, D, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None


# ── forward ───────────────────────────────────────────────────────────────────


class TestSOFTSForward:
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
            m = SOFTS(seq_len=96, pred_len=pred_len, enc_in=7, d_model=32)
            out = m(torch.randn(2, 96, 7))
            assert out.shape == (2, pred_len, 7), f"Failed for pred_len={pred_len}"

    def test_output_finite(self):
        m = _make_model()
        assert torch.isfinite(m(torch.randn(2, 96, 7))).all()

    def test_single_channel(self):
        m = SOFTS(seq_len=48, pred_len=12, enc_in=1, d_model=32)
        out = m(torch.randn(4, 48, 1))
        assert out.shape == (4, 12, 1)

    def test_many_channels(self):
        m = SOFTS(seq_len=48, pred_len=12, enc_in=137, d_model=32)
        out = m(torch.randn(2, 48, 137))
        assert out.shape == (2, 12, 137)

    def test_gradient_flows_to_input(self):
        m = _make_model(revin=False)
        x = torch.randn(2, 96, 7, requires_grad=True)
        out = m(x)
        out.sum().backward()
        assert x.grad is not None

    def test_gradient_flows_to_params(self):
        m = _make_model(revin=False)
        out = m(torch.randn(2, 96, 7))
        out.sum().backward()
        assert m.input_proj.weight.grad is not None

    def test_deterministic_eval(self):
        m = _make_model().eval()
        x = torch.randn(2, 96, 7)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_deeper_model(self):
        m = SOFTS(seq_len=96, pred_len=24, enc_in=7, d_model=64, e_layers=4)
        out = m(torch.randn(2, 96, 7))
        assert out.shape == (2, 24, 7)

    def test_batch_size_1(self):
        m = _make_model()
        out = m(torch.randn(1, 96, 7))
        assert out.shape == (1, 24, 7)

    def test_revin_normalizes_then_denormalizes(self):
        """With revin, the model output should track the scale of the input."""
        torch.manual_seed(42)
        m = _make_model(revin=True).eval()
        x_small = torch.randn(2, 96, 7) * 0.01
        x_large = torch.randn(2, 96, 7) * 100.0
        with torch.no_grad():
            out_small = m(x_small)
            out_large = m(x_large)
        # Large-scale input → large-scale output (RevIN denorm restores scale)
        assert out_large.abs().mean() > out_small.abs().mean()


# ── star-topology property ────────────────────────────────────────────────────


class TestSOFTSStarTopology:
    def test_o_n_complexity_does_not_grow_quadratically(self):
        """SOFTS core fusion is O(N); verify model params don't scale as N²."""
        N_small = 8
        N_large = 64
        m_small = SOFTS(seq_len=48, pred_len=12, enc_in=N_small, d_model=32)
        m_large = SOFTS(seq_len=48, pred_len=12, enc_in=N_large, d_model=32)
        n_small = sum(p.numel() for p in m_small.parameters())
        n_large = sum(p.numel() for p in m_large.parameters())
        # For a quadratic model the ratio would be (64/8)² = 64×
        # SOFTS is O(N) in the blocks (linear grows linearly with N via input/output proj)
        ratio = n_large / n_small
        assert ratio < 30, f"Param ratio {ratio:.1f} suggests super-linear N scaling"


# ── registry ──────────────────────────────────────────────────────────────────


class TestSOFTSRegistry:
    def test_importable_from_model_package(self):
        from torch_timeseries.model import SOFTS as S
        assert S is SOFTS

    def test_forecast_experiment_importable(self):
        from torch_timeseries.experiments.SOFTS import SOFTSForecast
        assert SOFTSForecast.model_type == "SOFTS"

    def test_all_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "AnomalyDetection", "Imputation", "UEAClassification"):
            cls = get_experiment_class("SOFTS", task)
            assert cls is not None
