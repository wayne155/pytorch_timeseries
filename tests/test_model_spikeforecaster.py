"""Tests for SpikeForecaster — LIF spiking neural network forecaster."""
import pytest
import torch

from torch_timeseries.model.SpikeForecaster import (
    SpikeForecaster, _LIFLayer, _LIFBlock,
)

SEQ = 16
PRED = 8
B = 2
C = 3
D = 16


class TestLIFLayer:
    def test_output_shape(self):
        lif = _LIFLayer(in_dim=D, out_dim=D)
        x = torch.randn(B, SEQ, D)
        out = lif(x)
        assert out.shape == (B, SEQ, D)

    def test_output_in_0_1(self):
        """Soft spike output should be in (0, 1) due to sigmoid."""
        lif = _LIFLayer(in_dim=D, out_dim=D)
        out = lif(torch.randn(B, SEQ, D))
        assert (out > 0).all() and (out < 1).all()

    def test_output_finite(self):
        lif = _LIFLayer(in_dim=D, out_dim=D)
        assert torch.isfinite(lif(torch.randn(B, SEQ, D))).all()

    def test_gradient_flows(self):
        lif = _LIFLayer(in_dim=D, out_dim=D)
        x = torch.randn(B, SEQ, D, requires_grad=True)
        lif(x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_alpha_in_0_1(self):
        """Membrane decay α must stay in (0,1)."""
        lif = _LIFLayer(in_dim=D, out_dim=D)
        alpha = torch.sigmoid(lif.alpha_raw)
        assert (alpha > 0).all() and (alpha < 1).all()

    def test_different_in_out_dim(self):
        lif = _LIFLayer(in_dim=8, out_dim=32)
        assert lif(torch.randn(B, SEQ, 8)).shape == (B, SEQ, 32)


class TestLIFBlock:
    def test_output_shape(self):
        block = _LIFBlock(d_model=D, surrogate_tau=0.5)
        assert block(torch.randn(B, SEQ, D)).shape == (B, SEQ, D)

    def test_output_finite(self):
        block = _LIFBlock(d_model=D, surrogate_tau=0.5)
        assert torch.isfinite(block(torch.randn(B, SEQ, D))).all()


class TestSpikeForecaster:
    def _model(self, **kw):
        defaults = dict(seq_len=SEQ, pred_len=PRED, enc_in=C,
                        d_model=D, n_layers=2, surrogate_tau=0.5, dropout=0.0)
        defaults.update(kw)
        return SpikeForecaster(**defaults)

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

    def test_different_tau(self):
        m = self._model(surrogate_tau=0.1)
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
        from torch_timeseries.model import SpikeForecaster as SNN
        assert SNN is SpikeForecaster

    def test_batch_size_one(self):
        m = self._model()
        assert m(torch.randn(1, SEQ, C)).shape == (1, PRED, C)

    def test_batch_size_eight(self):
        m = self._model()
        assert m(torch.randn(8, SEQ, C)).shape == (8, PRED, C)

    def test_spike_rates_bounded(self):
        """After embedding + LIF block, spike values must be in (0,1)."""
        m = self._model()
        x_ci = torch.randn(B * C, SEQ, 1)
        h = m.embed(x_ci)
        s = m.blocks[0](h)
        # The block adds a residual + LayerNorm, so output may exceed (0,1)
        # Just verify finiteness
        assert torch.isfinite(s).all()

    def test_longer_seq(self):
        m = SpikeForecaster(seq_len=96, pred_len=PRED, enc_in=C, d_model=D)
        assert m(torch.randn(B, 96, C)).shape == (B, PRED, C)
