"""Tests for EchoStateForecaster — Echo State Network (reservoir computing)."""
import pytest
import torch

from torch_timeseries.model.EchoStateForecaster import (
    EchoStateForecaster, _make_reservoir,
)

SEQ = 16
PRED = 8
B = 2
C = 3


class TestMakeReservoir:
    def test_shape(self):
        W = _make_reservoir(32, sparsity=0.9, spectral_radius=0.9)
        assert W.shape == (32, 32)

    def test_spectral_radius_bounded(self):
        W = _make_reservoir(32, sparsity=0.5, spectral_radius=0.9)
        eigvals = torch.linalg.eigvals(W)
        sr = eigvals.abs().max().item()
        assert sr <= 0.95   # allow tiny float rounding

    def test_sparsity(self):
        W = _make_reservoir(64, sparsity=0.9, spectral_radius=0.9)
        zero_frac = (W == 0).float().mean().item()
        assert zero_frac >= 0.8   # majority zeros

    def test_all_finite(self):
        W = _make_reservoir(32, sparsity=0.5, spectral_radius=0.95)
        assert torch.isfinite(W).all()


class TestEchoStateForecaster:
    def _model(self, **kw):
        defaults = dict(seq_len=SEQ, pred_len=PRED, enc_in=C,
                        d_reservoir=32, sparsity=0.5,
                        spectral_radius=0.9, dropout=0.0)
        defaults.update(kw)
        return EchoStateForecaster(**defaults)

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

    def test_leak_rate_partial(self):
        m = self._model(leak_rate=0.5)
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

    def test_reservoir_not_a_parameter(self):
        """W_res must be a buffer, not a trainable parameter."""
        m = self._model()
        param_names = {n for n, _ in m.named_parameters()}
        assert "W_res" not in param_names
        assert hasattr(m, "W_res")   # still accessible as buffer

    def test_trained_params_have_grad(self):
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

    def test_reservoir_fixed_across_forward_passes(self):
        """W_res must not change between forward calls."""
        m = self._model()
        W_before = m.W_res.clone()
        m(torch.randn(B, SEQ, C)).sum().backward()
        torch.optim.Adam(m.parameters(), lr=1e-3).step()
        assert torch.allclose(m.W_res, W_before)

    def test_importable_from_package(self):
        from torch_timeseries.model import EchoStateForecaster as ESN
        assert ESN is EchoStateForecaster

    def test_batch_size_one(self):
        m = self._model()
        assert m(torch.randn(1, SEQ, C)).shape == (1, PRED, C)

    def test_larger_reservoir(self):
        m = self._model(d_reservoir=128)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_high_sparsity(self):
        m = self._model(sparsity=0.95)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_longer_seq(self):
        m = EchoStateForecaster(seq_len=96, pred_len=PRED, enc_in=C, d_reservoir=32)
        assert m(torch.randn(B, 96, C)).shape == (B, PRED, C)
