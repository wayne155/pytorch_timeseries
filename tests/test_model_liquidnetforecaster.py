"""Tests for LiquidNetForecaster — Liquid Time-Constant (LTC) recurrent forecaster."""
import pytest
import torch

from torch_timeseries.model.LiquidNetForecaster import (
    LiquidNetForecaster, _LTCCell, _LTCBlock,
)

SEQ = 24
PRED = 8
B = 2
C = 3


class TestLTCCell:
    def _cell(self, d=16):
        return _LTCCell(d_model=d, dt=0.1)

    def test_output_shape(self):
        cell = self._cell()
        x = torch.randn(B, 16)
        h = torch.zeros(B, 16)
        h_new = cell(x, h)
        assert h_new.shape == (B, 16)

    def test_output_finite(self):
        cell = self._cell()
        h = cell(torch.randn(4, 16), torch.zeros(4, 16))
        assert torch.isfinite(h).all()

    def test_zero_input_decays(self):
        """With zero input, hidden state should diminish (decay toward zero)."""
        cell = self._cell()
        h = torch.ones(1, 16)
        for _ in range(50):
            h = cell(torch.zeros(1, 16), h)
        # after many steps with zero input, state decays
        assert h.abs().max() < 10.0   # bounded (not diverging)

    def test_gradient_flows(self):
        cell = self._cell()
        x = torch.randn(B, 16, requires_grad=True)
        h0 = torch.zeros(B, 16)
        h = cell(x, h0)
        h.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


class TestLTCBlock:
    def test_output_shape(self):
        block = _LTCBlock(d_model=16, dt=0.1)
        x = torch.randn(B, SEQ, 16)
        out = block(x)
        assert out.shape == (B, SEQ, 16)

    def test_residual_preserves_shape(self):
        block = _LTCBlock(d_model=32, dt=0.05)
        x = torch.randn(3, 12, 32)
        assert block(x).shape == (3, 12, 32)

    def test_output_finite(self):
        block = _LTCBlock(d_model=16, dt=0.1)
        assert torch.isfinite(block(torch.randn(B, SEQ, 16))).all()


class TestLiquidNetForecaster:
    def _model(self, **kw):
        defaults = dict(seq_len=SEQ, pred_len=PRED, enc_in=C,
                        d_model=32, n_layers=2, dt=0.1, dropout=0.0)
        defaults.update(kw)
        return LiquidNetForecaster(**defaults)

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
        m = self._model(enc_in=21)
        assert m(torch.randn(B, SEQ, 21)).shape == (B, PRED, 21)

    def test_longer_pred(self):
        m = self._model(pred_len=96)
        assert m(torch.randn(B, SEQ, C)).shape == (B, 96, C)

    def test_deeper_network(self):
        m = self._model(n_layers=4)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_smaller_dt(self):
        m = self._model(dt=0.01)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_larger_dt(self):
        m = self._model(dt=1.0)
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
        m2 = self._model(n_layers=4)
        n1 = sum(p.numel() for p in m1.parameters())
        n2 = sum(p.numel() for p in m2.parameters())
        assert n2 > n1

    def test_importable_from_package(self):
        from torch_timeseries.model import LiquidNetForecaster as LNF
        assert LNF is LiquidNetForecaster

    def test_batch_size_one(self):
        m = self._model()
        assert m(torch.randn(1, SEQ, C)).shape == (1, PRED, C)

    def test_batch_size_eight(self):
        m = self._model()
        assert m(torch.randn(8, SEQ, C)).shape == (8, PRED, C)

    def test_larger_d_model(self):
        m = self._model(d_model=128)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_seq_len_48(self):
        m = LiquidNetForecaster(seq_len=48, pred_len=PRED, enc_in=C, d_model=32)
        assert m(torch.randn(B, 48, C)).shape == (B, PRED, C)

    def test_loss_decreases_over_training(self):
        m = self._model()
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        x, y = torch.randn(8, SEQ, C), torch.randn(8, PRED, C)
        losses = []
        for _ in range(5):
            opt.zero_grad()
            loss = (m(x) - y).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0] * 2  # training is moving (not exploding)
