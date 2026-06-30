"""Tests for FlowTS — Time Series Generation via Rectified Flow."""
import pytest
import torch

from torch_timeseries.model.FlowTS import FlowTS, _RectifiedVelocityNet, _sinusoidal_emb

T  = 24   # seq_len
C  = 3    # n_features
B  = 4    # batch
D  = 16   # d_model


def _model(**kw):
    defaults = dict(seq_len=T, n_features=C, d_model=D, n_heads=2, n_layers=2, n_steps=5)
    defaults.update(kw)
    return FlowTS(**defaults)


class TestSinusoidalEmb:
    def test_shape(self):
        emb = _sinusoidal_emb(torch.rand(B), D)
        assert emb.shape == (B, D)

    def test_finite(self):
        assert torch.isfinite(_sinusoidal_emb(torch.tensor([0.0, 1.0]), D)).all()

    def test_different_for_different_t(self):
        t0 = _sinusoidal_emb(torch.zeros(1), D)
        t1 = _sinusoidal_emb(torch.ones(1), D)
        assert not torch.allclose(t0, t1)


class TestRectifiedVelocityNet:
    def _net(self):
        return _RectifiedVelocityNet(T, C, D, n_heads=2, n_layers=2)

    def test_output_shape(self):
        net = self._net()
        x_t = torch.randn(B, T, C)
        t   = torch.rand(B)
        out = net(x_t, t)
        assert out.shape == (B, T, C)

    def test_output_finite(self):
        net = self._net()
        assert torch.isfinite(net(torch.randn(B, T, C), torch.rand(B))).all()

    def test_t_zero_vs_one(self):
        net = self._net()
        x = torch.randn(1, T, C)
        v0 = net(x, torch.zeros(1))
        v1 = net(x, torch.ones(1))
        assert not torch.allclose(v0, v1)


class TestFlowTS:
    def test_loss_scalar(self):
        m = _model()
        loss = m.loss(torch.randn(B, T, C))
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_loss_backward(self):
        m = _model()
        m.loss(torch.randn(B, T, C)).backward()
        for p in m.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()

    def test_generate_shape(self):
        m = _model()
        out = m.generate(n=5, n_steps=3)
        assert out.shape == (5, T, C)

    def test_generate_finite(self):
        m = _model()
        out = m.generate(n=4, n_steps=3)
        assert torch.isfinite(out).all()

    def test_generate_returns_cpu(self):
        m = _model()
        out = m.generate(n=2, n_steps=2)
        assert out.device.type == "cpu"

    def test_generate_n_steps_override(self):
        m = FlowTS(seq_len=T, n_features=C, d_model=D, n_heads=2, n_layers=1, n_steps=10)
        out = m.generate(n=3, n_steps=3)
        assert out.shape == (3, T, C)

    def test_different_seqs_differ(self):
        m = _model()
        a = m.generate(n=2, n_steps=5)
        b = m.generate(n=2, n_steps=5)
        # Two separate calls from random noise should (almost certainly) differ
        assert not torch.allclose(a, b)

    def test_batch_size_one(self):
        m = _model()
        out = m.generate(n=1, n_steps=2)
        assert out.shape == (1, T, C)

    def test_n_features_multivariate(self):
        m = FlowTS(seq_len=T, n_features=7, d_model=D, n_heads=2, n_layers=1, n_steps=3)
        out = m.generate(n=2, n_steps=3)
        assert out.shape == (2, T, 7)
