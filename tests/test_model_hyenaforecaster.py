"""Tests for HyenaForecaster — position-conditioned long-convolution forecaster."""
import math
import pytest
import torch
import torch.nn as nn

from torch_timeseries.model.HyenaForecaster import (
    HyenaForecaster, _PosFilterMLP, _HyenaBlock,
)


SEQ = 32
PRED = 8
B = 2
C = 3


# ── _PosFilterMLP ─────────────────────────────────────────────────────────────

class TestPosFilterMLP:
    def test_output_shape(self):
        m = _PosFilterMLP(pos_freqs=8, filter_dim=32, d_model=16, seq_len=SEQ)
        h = m()
        assert h.shape == (SEQ, 16)

    def test_output_finite(self):
        m = _PosFilterMLP(pos_freqs=8, filter_dim=32, d_model=16, seq_len=SEQ)
        assert torch.isfinite(m()).all()

    def test_pos_feat_shape(self):
        K = 8
        m = _PosFilterMLP(pos_freqs=K, filter_dim=32, d_model=16, seq_len=SEQ)
        assert m.pos_feat.shape == (SEQ, 2 * K + 1)

    def test_pos_feat_not_parameter(self):
        """Positional features must be a buffer, not a learnable parameter."""
        m = _PosFilterMLP(pos_freqs=8, filter_dim=32, d_model=16, seq_len=SEQ)
        param_names = [n for n, _ in m.named_parameters()]
        assert "pos_feat" not in param_names

    def test_gradient_flows_through_mlp(self):
        m = _PosFilterMLP(pos_freqs=4, filter_dim=16, d_model=8, seq_len=SEQ)
        m().sum().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"No grad for {name}"

    def test_filter_varies_with_time(self):
        """Filter values at different time steps should differ."""
        m = _PosFilterMLP(pos_freqs=8, filter_dim=32, d_model=16, seq_len=SEQ)
        h = m()
        assert not torch.allclose(h[0], h[SEQ // 2])

    def test_filter_not_a_function_of_input(self):
        """Filter should be identical across batch calls (position-only)."""
        m = _PosFilterMLP(pos_freqs=8, filter_dim=32, d_model=16, seq_len=SEQ)
        h1 = m()
        h2 = m()
        assert torch.allclose(h1, h2)


# ── _HyenaBlock ───────────────────────────────────────────────────────────────

class TestHyenaBlock:
    def test_output_shape(self):
        block = _HyenaBlock(d_model=16, pos_freqs=8, filter_dim=32,
                            seq_len=SEQ, dropout=0.0)
        x = torch.randn(B, SEQ, 16)
        assert block(x).shape == (B, SEQ, 16)

    def test_output_finite(self):
        block = _HyenaBlock(d_model=16, pos_freqs=8, filter_dim=32,
                            seq_len=SEQ, dropout=0.0)
        x = torch.randn(B, SEQ, 16)
        assert torch.isfinite(block(x)).all()


# ── HyenaForecaster ───────────────────────────────────────────────────────────

class TestHyenaForecaster:
    def _model(self, **kw):
        defaults = dict(seq_len=SEQ, pred_len=PRED, enc_in=C,
                        d_model=16, n_layers=2, pos_freqs=4,
                        filter_dim=32, dropout=0.0)
        defaults.update(kw)
        return HyenaForecaster(**defaults)

    def test_output_shape(self):
        m = self._model()
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_output_finite(self):
        m = self._model()
        assert torch.isfinite(m(torch.randn(B, SEQ, C))).all()

    def test_no_revin(self):
        m = self._model(revin=False)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_single_channel(self):
        m = self._model(enc_in=1)
        assert m(torch.randn(B, SEQ, 1)).shape == (B, PRED, 1)

    def test_single_layer(self):
        m = self._model(n_layers=1)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_deeper_model(self):
        m = self._model(n_layers=4, d_model=32)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_larger_pred_len(self):
        m = self._model(pred_len=96)
        assert m(torch.randn(B, SEQ, C)).shape == (B, 96, C)

    def test_many_channels(self):
        m = self._model(enc_in=21)
        assert m(torch.randn(B, SEQ, 21)).shape == (B, PRED, 21)

    def test_more_pos_freqs(self):
        m = self._model(pos_freqs=32)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_eval_deterministic(self):
        m = self._model(dropout=0.1).eval()
        x = torch.randn(B, SEQ, C)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))

    def test_gradient_flows(self):
        m = self._model()
        x = torch.randn(B, SEQ, C, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_all_params_have_grad(self):
        m = self._model()
        m(torch.randn(B, SEQ, C)).sum().backward()
        for name, p in m.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_filter_is_position_based(self):
        """Filter MLP weights — not pos_feat — should be in named_parameters."""
        m = self._model()
        param_names = [n for n, _ in m.named_parameters()]
        assert any("filter_mlp.mlp" in n for n in param_names)
        assert all("pos_feat" not in n for n in param_names)

    def test_importable_from_package(self):
        from torch_timeseries.model import HyenaForecaster as Hyena
        assert Hyena is HyenaForecaster

    def test_training_step(self):
        m = self._model()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        x, y = torch.randn(4, SEQ, C), torch.randn(4, PRED, C)
        for _ in range(3):
            opt.zero_grad()
            loss = (m(x) - y).pow(2).mean()
            loss.backward()
            opt.step()
        assert torch.isfinite(loss)

    def test_fft_conv_used(self):
        """Forward pass must use FFT-based convolution (check block internals)."""
        import inspect
        from torch_timeseries.model.HyenaForecaster import _HyenaBlock
        src = inspect.getsource(_HyenaBlock.forward)
        assert "rfft" in src

    def test_larger_filter_dim_more_params(self):
        m_small = self._model(filter_dim=16)
        m_large = self._model(filter_dim=128)
        n_small = sum(p.numel() for p in m_small.parameters())
        n_large = sum(p.numel() for p in m_large.parameters())
        assert n_large > n_small
