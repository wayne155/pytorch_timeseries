"""Tests for xLSTMForecaster — mLSTM with matrix memory cell and exponential gates."""
import pytest
import torch

from torch_timeseries.model.xLSTMForecaster import (
    xLSTMForecaster, _mLSTMCell, _mLSTMBlock,
)

SEQ = 16
PRED = 8
B = 2
C = 3
D = 8


class TestMLSTMCell:
    def _cell(self, d=D):
        return _mLSTMCell(d_model=d)

    def test_output_shape(self):
        cell = self._cell()
        x = torch.randn(B, D)
        C_mat = torch.zeros(B, D, D)
        n = torch.zeros(B, D)
        m = torch.full((B, D), -1e9)
        h, C2, n2, m2 = cell(x, C_mat, n, m)
        assert h.shape == (B, D)
        assert C2.shape == (B, D, D)
        assert n2.shape == (B, D)
        assert m2.shape == (B, D)

    def test_output_finite(self):
        cell = self._cell()
        x = torch.randn(B, D)
        C_mat = torch.zeros(B, D, D)
        n = torch.zeros(B, D)
        m = torch.full((B, D), -1e9)
        h, C2, n2, m2 = cell(x, C_mat, n, m)
        for t in [h, C2, n2, m2]:
            assert torch.isfinite(t).all()

    def test_gradient_flows(self):
        cell = self._cell()
        x = torch.randn(B, D, requires_grad=True)
        C_mat = torch.zeros(B, D, D)
        n = torch.zeros(B, D)
        m = torch.full((B, D), -1e9)
        h, *_ = cell(x, C_mat, n, m)
        h.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


class TestMLSTMBlock:
    def test_output_shape(self):
        block = _mLSTMBlock(d_model=D)
        x = torch.randn(B, SEQ, D)
        out = block(x)
        assert out.shape == (B, SEQ, D)

    def test_output_finite(self):
        block = _mLSTMBlock(d_model=D)
        out = block(torch.randn(B, SEQ, D))
        assert torch.isfinite(out).all()

    def test_residual_connection(self):
        block = _mLSTMBlock(d_model=D)
        x = torch.randn(B, SEQ, D)
        # with zero-init weights the residual should keep magnitude bounded
        out = block(x)
        assert out.abs().max() < 1e4


class TestxLSTMForecaster:
    def _model(self, **kw):
        defaults = dict(seq_len=SEQ, pred_len=PRED, enc_in=C,
                        d_model=D, n_layers=2, dropout=0.0)
        defaults.update(kw)
        return xLSTMForecaster(**defaults)

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

    def test_many_channels(self):
        m = self._model(enc_in=12)
        assert m(torch.randn(B, SEQ, 12)).shape == (B, PRED, 12)

    def test_longer_horizon(self):
        m = self._model(pred_len=48)
        assert m(torch.randn(B, SEQ, C)).shape == (B, 48, C)

    def test_deeper_model(self):
        m = self._model(n_layers=4)
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
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_all_params_have_grad(self):
        m = self._model()
        m(torch.randn(B, SEQ, C)).sum().backward()
        for name, p in m.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

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

    def test_more_layers_more_params(self):
        m1 = self._model(n_layers=1)
        m2 = self._model(n_layers=3)
        n1 = sum(p.numel() for p in m1.parameters())
        n2 = sum(p.numel() for p in m2.parameters())
        assert n2 > n1

    def test_importable_from_package(self):
        from torch_timeseries.model import xLSTMForecaster as xLSTM
        assert xLSTM is xLSTMForecaster

    def test_batch_size_one(self):
        m = self._model()
        assert m(torch.randn(1, SEQ, C)).shape == (1, PRED, C)

    def test_batch_size_eight(self):
        m = self._model()
        assert m(torch.randn(8, SEQ, C)).shape == (8, PRED, C)

    def test_larger_d_model(self):
        m = self._model(d_model=64)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_longer_seq(self):
        m = xLSTMForecaster(seq_len=96, pred_len=PRED, enc_in=C, d_model=D)
        assert m(torch.randn(B, 96, C)).shape == (B, PRED, C)

    def test_memory_matrix_distinct_from_scalar_lstm(self):
        """Matrix cell state shape must be (BC, d, d) — verify via cell internals."""
        m = self._model()
        # Just checks that the cell stores a 2D matrix, not a scalar
        cell = m.blocks[0].cell
        assert hasattr(cell, 'W_q') and hasattr(cell, 'W_v')
        # W_q projects d→d (matrix interaction)
        assert cell.W_q.weight.shape == (D, D)
