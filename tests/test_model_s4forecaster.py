"""Tests for S4Forecaster — diagonal structured state space forecaster."""
import math
import pytest
import torch
import torch.nn as nn

from torch_timeseries.model.S4Forecaster import (
    S4Forecaster, _S4DKernel, _S4DBlock, _fft_conv,
)


SEQ = 32
PRED = 8
B = 2
C = 3


# ── _fft_conv ─────────────────────────────────────────────────────────────────

class TestFFTConv:
    def test_output_shape(self):
        x = torch.randn(B, 4, SEQ)
        k = torch.randn(4, SEQ)
        y = _fft_conv(x, k)
        assert y.shape == (B, 4, SEQ)

    def test_identity_kernel(self):
        """Kernel [1, 0, 0, ...] is the identity convolution."""
        x = torch.randn(2, 1, SEQ)
        k = torch.zeros(1, SEQ)
        k[0, 0] = 1.0
        y = _fft_conv(x, k)
        assert torch.allclose(y, x, atol=1e-5)

    def test_output_finite(self):
        x = torch.randn(B, 3, SEQ)
        k = torch.randn(3, SEQ)
        assert torch.isfinite(_fft_conv(x, k)).all()


# ── _S4DKernel ────────────────────────────────────────────────────────────────

class TestS4DKernel:
    def test_kernel_shape(self):
        kern = _S4DKernel(d_model=8, d_state=16, seq_len=SEQ)
        k = kern()
        assert k.shape == (8, SEQ)

    def test_kernel_finite(self):
        kern = _S4DKernel(d_model=8, d_state=16, seq_len=SEQ)
        assert torch.isfinite(kern()).all()

    def test_decay_positive(self):
        """Eigenvalue magnitudes must be < 1 (stable SSM)."""
        import torch.nn.functional as F
        kern = _S4DKernel(d_model=4, d_state=8, seq_len=SEQ)
        decay = F.softplus(kern.log_decay)
        assert (decay > 0).all()

    def test_kernel_decays_toward_zero(self):
        """With large decay, kernel values at t >> 0 should be small."""
        kern = _S4DKernel(d_model=2, d_state=4, seq_len=SEQ)
        with torch.no_grad():
            kern.log_decay.fill_(5.0)  # large decay ≈ fast decay
            kern.phase.fill_(0.0)
            kern.B.fill_(1.0)
            kern.C.fill_(1.0)
        k = kern()
        assert k[:, -1].abs().max() < k[:, 0].abs().max()

    def test_gradient_flows(self):
        kern = _S4DKernel(d_model=4, d_state=8, seq_len=SEQ)
        kern().sum().backward()
        for name, p in kern.named_parameters():
            assert p.grad is not None, f"No grad for {name}"


# ── _S4DBlock ─────────────────────────────────────────────────────────────────

class TestS4DBlock:
    def test_output_shape(self):
        block = _S4DBlock(d_model=16, d_state=8, seq_len=SEQ, mlp_mult=2, dropout=0.0)
        x = torch.randn(B, SEQ, 16)
        assert block(x).shape == (B, SEQ, 16)

    def test_output_finite(self):
        block = _S4DBlock(d_model=16, d_state=8, seq_len=SEQ, mlp_mult=2, dropout=0.0)
        x = torch.randn(B, SEQ, 16)
        assert torch.isfinite(block(x)).all()


# ── S4Forecaster ──────────────────────────────────────────────────────────────

class TestS4Forecaster:
    def _model(self, **kw):
        defaults = dict(seq_len=SEQ, pred_len=PRED, enc_in=C,
                        d_model=16, d_state=8, n_layers=2, mlp_mult=2, dropout=0.0)
        defaults.update(kw)
        return S4Forecaster(**defaults)

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
        m = self._model(n_layers=4, d_model=32, d_state=16)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_larger_pred_len(self):
        m = self._model(pred_len=96)
        assert m(torch.randn(B, SEQ, C)).shape == (B, 96, C)

    def test_many_channels(self):
        m = self._model(enc_in=21)
        assert m(torch.randn(B, SEQ, 21)).shape == (B, PRED, 21)

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

    def test_fft_based_not_sequential(self):
        """S4D must use FFT convolution, not a sequential scan (no RNN-style loop)."""
        import inspect
        from torch_timeseries.model.S4Forecaster import _S4DBlock
        src = inspect.getsource(_S4DBlock.forward)
        assert "fft_conv" in src
        assert "for t in range" not in src

    def test_importable_from_package(self):
        from torch_timeseries.model import S4Forecaster as S4
        assert S4 is S4Forecaster

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

    def test_larger_d_state_more_params(self):
        m_small = self._model(d_state=4)
        m_large = self._model(d_state=32)
        n_small = sum(p.numel() for p in m_small.parameters())
        n_large = sum(p.numel() for p in m_large.parameters())
        assert n_large > n_small
