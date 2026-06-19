"""Tests for GatedMLPForecaster — gMLP with Spatial Gating Units."""
import pytest
import torch
import torch.nn as nn

from torch_timeseries.model.GatedMLPForecaster import (
    GatedMLPForecaster, _SGU, _gMLPBlock,
)


SEQ = 24
PRED = 8
B = 2
C = 3


# ── _SGU ──────────────────────────────────────────────────────────────────────

class TestSGU:
    def test_output_shape(self):
        sgu = _SGU(seq_len=SEQ, d_ffn=32, dropout=0.0)
        u = torch.randn(B, SEQ, 32)
        v = torch.randn(B, SEQ, 32)
        out = sgu(u, v)
        assert out.shape == (B, SEQ, 32)

    def test_output_finite(self):
        sgu = _SGU(seq_len=SEQ, d_ffn=32, dropout=0.0)
        out = sgu(torch.randn(B, SEQ, 32), torch.randn(B, SEQ, 32))
        assert torch.isfinite(out).all()

    def test_gradient_flows(self):
        sgu = _SGU(seq_len=SEQ, d_ffn=16, dropout=0.0)
        u = torch.randn(B, SEQ, 16, requires_grad=True)
        v = torch.randn(B, SEQ, 16, requires_grad=True)
        sgu(u, v).sum().backward()
        assert u.grad is not None
        assert v.grad is not None

    def test_spatial_weight_is_parameter(self):
        sgu = _SGU(seq_len=SEQ, d_ffn=16, dropout=0.0)
        assert isinstance(sgu.W_s.weight, nn.Parameter)
        assert sgu.W_s.weight.shape == (SEQ, SEQ)


# ── _gMLPBlock ────────────────────────────────────────────────────────────────

class TestgMLPBlock:
    def test_output_shape(self):
        block = _gMLPBlock(d_model=16, d_ffn=32, seq_len=SEQ, dropout=0.0)
        x = torch.randn(B, SEQ, 16)
        assert block(x).shape == (B, SEQ, 16)

    def test_output_finite(self):
        block = _gMLPBlock(d_model=16, d_ffn=32, seq_len=SEQ, dropout=0.0)
        assert torch.isfinite(block(torch.randn(B, SEQ, 16))).all()

    def test_residual_connection(self):
        """With near-zero init, output should be close to input."""
        block = _gMLPBlock(d_model=8, d_ffn=8, seq_len=SEQ, dropout=0.0)
        nn.init.zeros_(block.proj_out.weight)
        nn.init.zeros_(block.proj_out.bias)
        x = torch.randn(B, SEQ, 8)
        with torch.no_grad():
            y = block(x)
        # Output ≈ input when proj_out is zero
        assert torch.allclose(y, x, atol=1e-5)


# ── GatedMLPForecaster ────────────────────────────────────────────────────────

class TestGatedMLPForecaster:
    def _model(self, **kw):
        defaults = dict(seq_len=SEQ, pred_len=PRED, enc_in=C,
                        d_model=16, d_ffn=32, n_layers=2, dropout=0.0)
        defaults.update(kw)
        return GatedMLPForecaster(**defaults)

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

    def test_larger_pred_len(self):
        m = self._model(pred_len=96)
        assert m(torch.randn(B, SEQ, C)).shape == (B, 96, C)

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
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_all_params_have_grad(self):
        m = self._model()
        m(torch.randn(B, SEQ, C)).sum().backward()
        for name, p in m.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_spatial_weight_shape(self):
        """SGU must have a T×T spatial weight matrix."""
        m = self._model()
        block = m.blocks[0]
        W_s = block.sgu.W_s
        assert W_s.weight.shape == (SEQ, SEQ)

    def test_no_attention(self):
        """gMLP must not use nn.MultiheadAttention."""
        m = self._model()
        for name, module in m.named_modules():
            assert not isinstance(module, nn.MultiheadAttention), \
                f"Found attention module at {name}"

    def test_importable_from_package(self):
        from torch_timeseries.model import GatedMLPForecaster as GMLP
        assert GMLP is GatedMLPForecaster

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

    def test_larger_ffn_more_params(self):
        m_small = self._model(d_ffn=16)
        m_large = self._model(d_ffn=256)
        n_small = sum(p.numel() for p in m_small.parameters())
        n_large = sum(p.numel() for p in m_large.parameters())
        assert n_large > n_small
