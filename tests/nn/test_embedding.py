"""Tests for torch_timeseries.nn embedding modules."""
import pytest
import torch

from torch_timeseries.nn import (
    DataEmbedding,
    DataEmbedding_wo_pos,
    DataEmbedding_inverted,
    PatchEmbedding,
    PositionalEmbedding,
    TokenEmbedding,
    FixedEmbedding,
    TemporalEmbedding,
    TimeFeatureEmbedding,
)

B, L, C, D = 4, 96, 7, 64
# x_mark: (B, L, 4) — [month, day, weekday, hour] as integers in the right range
# TemporalEmbedding expects columns: [month(0-12), day(0-31), weekday(0-6), hour(0-23)]
def _make_x_mark(batch=B, seq=L):
    mark = torch.zeros(batch, seq, 4, dtype=torch.long)
    mark[:, :, 0] = 1   # month  in [0, 12]
    mark[:, :, 1] = 1   # day    in [0, 31]
    mark[:, :, 2] = 1   # weekday in [0, 6]
    mark[:, :, 3] = 1   # hour   in [0, 23]
    return mark.float()


# ── PositionalEmbedding ───────────────────────────────────────────────────────

class TestPositionalEmbedding:
    def test_output_shape(self):
        pe = PositionalEmbedding(d_model=D)
        x = torch.randn(B, L, D)
        out = pe(x)
        assert out.shape == (1, L, D)

    def test_no_grad(self):
        pe = PositionalEmbedding(d_model=D)
        x = torch.randn(B, L, D)
        out = pe(x)
        assert not out.requires_grad

    def test_shorter_seq(self):
        pe = PositionalEmbedding(d_model=D, max_len=500)
        x = torch.randn(B, 48, D)
        out = pe(x)
        assert out.shape == (1, 48, D)

    def test_no_nan(self):
        pe = PositionalEmbedding(d_model=D)
        x = torch.randn(B, L, D)
        assert not torch.isnan(pe(x)).any()


# ── TokenEmbedding ────────────────────────────────────────────────────────────

class TestTokenEmbedding:
    def test_output_shape(self):
        te = TokenEmbedding(c_in=C, d_model=D)
        x = torch.randn(B, L, C)
        out = te(x)
        assert out.shape == (B, L, D)

    def test_gradients_flow(self):
        te = TokenEmbedding(c_in=C, d_model=D)
        x = torch.randn(B, L, C, requires_grad=True)
        te(x).sum().backward()
        assert x.grad is not None

    def test_no_nan(self):
        te = TokenEmbedding(c_in=C, d_model=D)
        assert not torch.isnan(te(torch.randn(B, L, C))).any()


# ── FixedEmbedding ────────────────────────────────────────────────────────────

class TestFixedEmbedding:
    def test_output_shape(self):
        fe = FixedEmbedding(c_in=24, d_model=D)
        x = torch.randint(0, 24, (B, L))
        out = fe(x)
        assert out.shape == (B, L, D)

    def test_weights_not_trained(self):
        fe = FixedEmbedding(c_in=24, d_model=D)
        assert not fe.emb.weight.requires_grad

    def test_no_nan(self):
        fe = FixedEmbedding(c_in=24, d_model=D)
        x = torch.randint(0, 24, (B, L))
        assert not torch.isnan(fe(x)).any()


# ── TemporalEmbedding ─────────────────────────────────────────────────────────

class TestTemporalEmbedding:
    def test_output_shape_fixed(self):
        te = TemporalEmbedding(d_model=D, embed_type='fixed', freq='h')
        x_mark = _make_x_mark()
        out = te(x_mark)
        assert out.shape == (B, L, D)

    def test_output_shape_learnable(self):
        te = TemporalEmbedding(d_model=D, embed_type='learned', freq='h')
        x_mark = _make_x_mark()
        out = te(x_mark)
        assert out.shape == (B, L, D)

    def test_no_nan(self):
        te = TemporalEmbedding(d_model=D, embed_type='fixed', freq='h')
        assert not torch.isnan(te(_make_x_mark())).any()


# ── TimeFeatureEmbedding ──────────────────────────────────────────────────────

class TestTimeFeatureEmbedding:
    def test_output_shape_hourly(self):
        tfe = TimeFeatureEmbedding(d_model=D, freq='h')
        x = torch.randn(B, L, 4)   # hourly has 4 time features
        out = tfe(x)
        assert out.shape == (B, L, D)

    def test_output_shape_daily(self):
        tfe = TimeFeatureEmbedding(d_model=D, freq='d')
        x = torch.randn(B, L, 3)   # daily has 3 time features
        out = tfe(x)
        assert out.shape == (B, L, D)

    def test_gradients_flow(self):
        tfe = TimeFeatureEmbedding(d_model=D, freq='h')
        x = torch.randn(B, L, 4, requires_grad=True)
        tfe(x).sum().backward()
        assert x.grad is not None


# ── DataEmbedding ─────────────────────────────────────────────────────────────

class TestDataEmbedding:
    def test_output_shape_with_mark(self):
        emb = DataEmbedding(c_in=C, d_model=D, embed_type='fixed', freq='h', dropout=0.0)
        x = torch.randn(B, L, C)
        x_mark = _make_x_mark()
        out = emb(x, x_mark)
        assert out.shape == (B, L, D)

    def test_output_shape_without_mark(self):
        emb = DataEmbedding(c_in=C, d_model=D, embed_type='fixed', freq='h',
                             dropout=0.0, time_embed=True)
        x = torch.randn(B, L, C)
        out = emb(x, None)
        assert out.shape == (B, L, D)

    def test_no_time_embed(self):
        emb = DataEmbedding(c_in=C, d_model=D, time_embed=False, dropout=0.0)
        x = torch.randn(B, L, C)
        out = emb(x, None)
        assert out.shape == (B, L, D)

    def test_no_nan(self):
        emb = DataEmbedding(c_in=C, d_model=D, dropout=0.0)
        assert not torch.isnan(emb(torch.randn(B, L, C), _make_x_mark())).any()

    def test_gradients_flow(self):
        emb = DataEmbedding(c_in=C, d_model=D, dropout=0.0)
        x = torch.randn(B, L, C, requires_grad=True)
        out = emb(x, _make_x_mark())
        out.sum().backward()
        assert x.grad is not None


# ── DataEmbedding_wo_pos ──────────────────────────────────────────────────────

class TestDataEmbeddingWoPos:
    def test_output_shape_with_mark(self):
        emb = DataEmbedding_wo_pos(c_in=C, d_model=D, dropout=0.0)
        out = emb(torch.randn(B, L, C), _make_x_mark())
        assert out.shape == (B, L, D)

    def test_output_shape_without_mark(self):
        emb = DataEmbedding_wo_pos(c_in=C, d_model=D, dropout=0.0)
        out = emb(torch.randn(B, L, C), None)
        assert out.shape == (B, L, D)

    def test_no_nan(self):
        emb = DataEmbedding_wo_pos(c_in=C, d_model=D, dropout=0.0)
        assert not torch.isnan(emb(torch.randn(B, L, C), None)).any()


# ── DataEmbedding_inverted ────────────────────────────────────────────────────

class TestDataEmbeddingInverted:
    def test_output_shape_no_mark(self):
        emb = DataEmbedding_inverted(c_in=L, d_model=D, dropout=0.0)
        x = torch.randn(B, L, C)
        out = emb(x, None)
        assert out.shape == (B, C, D)

    def test_no_nan(self):
        emb = DataEmbedding_inverted(c_in=L, d_model=D, dropout=0.0)
        assert not torch.isnan(emb(torch.randn(B, L, C), None)).any()

    def test_gradients_flow(self):
        emb = DataEmbedding_inverted(c_in=L, d_model=D, dropout=0.0)
        x = torch.randn(B, L, C, requires_grad=True)
        emb(x, None).sum().backward()
        assert x.grad is not None


# ── PatchEmbedding ────────────────────────────────────────────────────────────

class TestPatchEmbedding:
    def test_output_shape(self):
        patch_len, stride, padding = 16, 8, 8
        emb = PatchEmbedding(d_model=D, patch_len=patch_len, stride=stride,
                              padding=padding, dropout=0.0)
        x = torch.randn(B, C, L)    # (B, C, L) — channels-first
        out, n_vars = emb(x)
        assert n_vars == C
        assert out.dim() == 3
        assert out.shape[-1] == D

    def test_no_nan(self):
        patch_len, stride, padding = 16, 8, 8
        emb = PatchEmbedding(d_model=D, patch_len=patch_len, stride=stride,
                              padding=padding, dropout=0.0)
        out, _ = emb(torch.randn(B, C, L))
        assert not torch.isnan(out).any()
