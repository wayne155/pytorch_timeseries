"""Tests for SincNetForecaster — learnable sinc bandpass filter forecaster."""
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.model.SincNetForecaster import (
    SincNetForecaster, _SincConv1d,
)


SEQ = 48
PRED = 12
B = 2
C = 3


# ── _SincConv1d ───────────────────────────────────────────────────────────────

class TestSincConv1d:
    def test_output_shape(self):
        layer = _SincConv1d(out_channels=16, kernel_size=25)
        x = torch.randn(B, 1, SEQ)
        y = layer(x)
        assert y.shape == (B, 16, SEQ)

    def test_output_finite(self):
        layer = _SincConv1d(out_channels=16, kernel_size=25)
        x = torch.randn(B, 1, SEQ)
        assert torch.isfinite(layer(x)).all()

    def test_cutoffs_positive(self):
        """f1 and f2 must always be strictly positive."""
        layer = _SincConv1d(out_channels=8, kernel_size=11)
        f1 = layer.alpha1.abs().clamp(0.001, 0.499) * 0.5
        df = layer.alpha2.abs().clamp(0.001, 0.499) * 0.5
        f2 = (f1 + df).clamp(max=0.499)
        assert (f1 > 0).all()
        assert (f2 > 0).all()

    def test_f2_greater_than_f1(self):
        """f2 must be at least f1 (bandpass requires low < high)."""
        layer = _SincConv1d(out_channels=8, kernel_size=11)
        f1 = layer.alpha1.abs().clamp(0.001, 0.499) * 0.5
        df = layer.alpha2.abs().clamp(0.001, 0.499) * 0.5
        f2 = (f1 + df).clamp(max=0.499)
        assert (f2 >= f1).all()

    def test_gradient_flows(self):
        layer = _SincConv1d(out_channels=8, kernel_size=11)
        x = torch.randn(2, 1, SEQ, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None
        assert layer.alpha1.grad is not None
        assert layer.alpha2.grad is not None

    def test_odd_kernel_required(self):
        with pytest.raises(AssertionError):
            _SincConv1d(out_channels=4, kernel_size=10)

    def test_different_filters_produce_different_output(self):
        """Filters initialised at different frequencies should differ."""
        layer = _SincConv1d(out_channels=16, kernel_size=25)
        filters = layer._compute_filters()   # (16, 1, 25)
        # Not all identical
        assert not torch.allclose(filters[0], filters[8])

    def test_filters_not_a_parameter(self):
        """Filters are computed, not stored — alpha1/alpha2 are the parameters."""
        layer = _SincConv1d(out_channels=8, kernel_size=11)
        param_names = [n for n, _ in layer.named_parameters()]
        assert "alpha1" in param_names
        assert "alpha2" in param_names


# ── SincNetForecaster ─────────────────────────────────────────────────────────

class TestSincNetForecaster:
    def _model(self, **kw):
        defaults = dict(seq_len=SEQ, pred_len=PRED, enc_in=C,
                        n_filters=16, kernel_size=11, n_conv_layers=1, dropout=0.0)
        defaults.update(kw)
        return SincNetForecaster(**defaults)

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

    def test_more_filters(self):
        m = self._model(n_filters=64)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_even_kernel_auto_fixed(self):
        """Even kernel_size should be incremented to odd automatically."""
        m = SincNetForecaster(seq_len=SEQ, pred_len=PRED, enc_in=C,
                              n_filters=8, kernel_size=10, n_conv_layers=0)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_zero_conv_layers(self):
        m = self._model(n_conv_layers=0)
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

    def test_cutoff_params_have_grad(self):
        m = self._model()
        m(torch.randn(B, SEQ, C)).sum().backward()
        assert m.sinc_conv.alpha1.grad is not None
        assert m.sinc_conv.alpha2.grad is not None

    def test_all_params_have_grad(self):
        m = self._model()
        m(torch.randn(B, SEQ, C)).sum().backward()
        for name, p in m.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_importable_from_package(self):
        from torch_timeseries.model import SincNetForecaster as SNC
        assert SNC is SincNetForecaster

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

    def test_more_filters_more_params(self):
        m_small = self._model(n_filters=8)
        m_large = self._model(n_filters=64)
        n_small = sum(p.numel() for p in m_small.parameters())
        n_large = sum(p.numel() for p in m_large.parameters())
        assert n_large > n_small
