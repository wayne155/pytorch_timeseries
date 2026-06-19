"""Tests for NystromForecaster — Nyström approximation of softmax attention."""
import pytest
import torch

from torch_timeseries.model.NystromForecaster import (
    NystromForecaster,
    _NystromAttentionBlock,
    _NystromBlock,
    _iterative_pinv,
)

SEQ = 16
PRED = 8
B = 2
C = 3
D = 16
H = 4
M = 4   # n_landmarks


class TestIterativePinv:
    def test_pseudoinverse_property(self):
        """A · pinv(A) · A ≈ A for a PSD matrix."""
        A = torch.eye(4)
        Ainv = _iterative_pinv(A)
        assert torch.allclose(A @ Ainv @ A, A, atol=1e-3)

    def test_output_shape(self):
        A = torch.randn(2, 4, 4)
        out = _iterative_pinv(A)
        assert out.shape == (2, 4, 4)


class TestNystromAttentionBlock:
    def test_output_shape(self):
        block = _NystromAttentionBlock(d_model=D, n_heads=H, n_landmarks=M, dropout=0.0)
        assert block(torch.randn(B, SEQ, D)).shape == (B, SEQ, D)

    def test_output_finite(self):
        block = _NystromAttentionBlock(d_model=D, n_heads=H, n_landmarks=M, dropout=0.0)
        assert torch.isfinite(block(torch.randn(B, SEQ, D))).all()

    def test_gradient_flows(self):
        block = _NystromAttentionBlock(d_model=D, n_heads=H, n_landmarks=M, dropout=0.0)
        x = torch.randn(B, SEQ, D, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_single_timestep(self):
        block = _NystromAttentionBlock(d_model=D, n_heads=H, n_landmarks=1, dropout=0.0)
        assert block(torch.randn(B, 1, D)).shape == (B, 1, D)


class TestNystromBlock:
    def test_output_shape(self):
        block = _NystromBlock(d_model=D, n_heads=H, n_landmarks=M, d_ffn=64, dropout=0.0)
        assert block(torch.randn(B, SEQ, D)).shape == (B, SEQ, D)

    def test_output_finite(self):
        block = _NystromBlock(d_model=D, n_heads=H, n_landmarks=M, d_ffn=64, dropout=0.0)
        assert torch.isfinite(block(torch.randn(B, SEQ, D))).all()


class TestNystromForecaster:
    def _model(self, **kw):
        defaults = dict(
            seq_len=SEQ, pred_len=PRED, enc_in=C,
            d_model=D, n_heads=H, n_landmarks=M, d_ffn=64, n_layers=2, dropout=0.0,
        )
        defaults.update(kw)
        return NystromForecaster(**defaults)

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

    def test_single_landmark(self):
        m = self._model(n_landmarks=1)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_more_landmarks(self):
        m = self._model(n_landmarks=SEQ)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_single_layer(self):
        m = self._model(n_layers=1)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

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
        from torch_timeseries.model import NystromForecaster as NF
        assert NF is NystromForecaster

    def test_batch_size_one(self):
        m = self._model()
        assert m(torch.randn(1, SEQ, C)).shape == (1, PRED, C)

    def test_longer_seq(self):
        m = NystromForecaster(
            seq_len=96, pred_len=PRED, enc_in=C,
            d_model=D, n_heads=H, n_landmarks=M, d_ffn=64
        )
        assert m(torch.randn(B, 96, C)).shape == (B, PRED, C)
