"""Tests for TSFlowForecaster — Flow Matching with GP Priors."""
import pytest
import torch

from torch_timeseries.model.TSFlowForecaster import (
    TSFlowForecaster,
    _VelocityTransformer,
    _se_kernel,
    _sinusoidal_emb,
)

SEQ  = 16
PRED = 8
B    = 2
C    = 3
D    = 32


def _model(**kw):
    defaults = dict(seq_len=SEQ, pred_len=PRED, enc_in=C, d_model=D,
                    flow_layers=2, n_steps=4, num_samples=3)
    defaults.update(kw)
    return TSFlowForecaster(**defaults)


class TestSinusoidalEmb:
    def test_shape(self):
        t = torch.rand(B)
        emb = _sinusoidal_emb(t, D)
        assert emb.shape == (B, D)

    def test_finite(self):
        emb = _sinusoidal_emb(torch.tensor([0.0, 0.5, 1.0]), D)
        assert torch.isfinite(emb).all()


class TestSEKernel:
    def test_shape(self):
        K = _se_kernel(PRED, 2.0, torch.device("cpu"))
        assert K.shape == (PRED, PRED)

    def test_symmetric_positive_definite(self):
        K = _se_kernel(PRED, 2.0, torch.device("cpu"))
        assert torch.allclose(K, K.T, atol=1e-6)
        eigs = torch.linalg.eigvalsh(K)
        assert (eigs > 0).all()

    def test_diagonal_ones(self):
        K = _se_kernel(PRED, 2.0, torch.device("cpu"))
        assert torch.allclose(K.diag(), torch.ones(PRED))


class TestVelocityTransformer:
    def _net(self):
        return _VelocityTransformer(PRED, C, D, D, n_layers=2)

    def test_output_shape(self):
        net = self._net()
        x_t = torch.randn(B, PRED, C)
        t   = torch.rand(B)
        ctx = torch.randn(B, D)
        out = net(x_t, t, ctx)
        assert out.shape == (B, PRED, C)

    def test_output_finite(self):
        net = self._net()
        out = net(torch.randn(B, PRED, C), torch.rand(B), torch.randn(B, D))
        assert torch.isfinite(out).all()


class TestTSFlowForecaster:
    def test_forward_shape(self):
        m = _model()
        x = torch.randn(B, SEQ, C)
        assert m(x).shape == (B, PRED, C)

    def test_forward_finite(self):
        m = _model()
        assert torch.isfinite(m(torch.randn(B, SEQ, C))).all()

    def test_flow_loss_scalar(self):
        m = _model()
        loss = m.flow_loss(torch.randn(B, SEQ, C), torch.randn(B, PRED, C))
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_sample_shape(self):
        m = _model()
        s = m.sample(torch.randn(B, SEQ, C), num_samples=4)
        assert s.shape == (B, PRED, C, 4)

    def test_sample_finite(self):
        m = _model()
        s = m.sample(torch.randn(B, SEQ, C), num_samples=2)
        assert torch.isfinite(s).all()

    def test_gp_buffer_registered(self):
        m = _model()
        assert hasattr(m, "gp_L")
        assert m.gp_L.shape == (PRED, PRED)

    def test_gp_sample_shape(self):
        m = _model()
        x0 = m._gp_sample(B)
        assert x0.shape == (B, PRED, C)

    def test_gp_sample_finite(self):
        m = _model()
        x0 = m._gp_sample(B)
        assert torch.isfinite(x0).all()

    def test_loss_backward(self):
        m = _model()
        loss = m.flow_loss(torch.randn(B, SEQ, C), torch.randn(B, PRED, C))
        loss.backward()
        for p in m.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()

    def test_different_n_steps(self):
        m = TSFlowForecaster(seq_len=SEQ, pred_len=PRED, enc_in=C,
                             d_model=D, n_steps=10, num_samples=2)
        s = m.sample(torch.randn(B, SEQ, C), num_samples=2)
        assert s.shape == (B, PRED, C, 2)

    def test_num_samples_default(self):
        m = TSFlowForecaster(seq_len=SEQ, pred_len=PRED, enc_in=C,
                             d_model=D, num_samples=5, n_steps=3)
        s = m.sample(torch.randn(B, SEQ, C))
        assert s.shape[3] == 5

    def test_gp_length_scale_affects_buffer(self):
        m1 = TSFlowForecaster(seq_len=SEQ, pred_len=PRED, enc_in=C, d_model=D,
                              gp_length_scale=1.0, n_steps=2)
        m2 = TSFlowForecaster(seq_len=SEQ, pred_len=PRED, enc_in=C, d_model=D,
                              gp_length_scale=5.0, n_steps=2)
        assert not torch.allclose(m1.gp_L, m2.gp_L)
