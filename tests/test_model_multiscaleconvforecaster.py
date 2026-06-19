"""Tests for MultiscaleConvForecaster — Inception multi-scale temporal CNN."""
import pytest
import torch
import torch.nn as nn

from torch_timeseries.model.MultiscaleConvForecaster import (
    MultiscaleConvForecaster, _InceptionBlock,
)


SEQ = 32
PRED = 8
B = 2
C = 3


# ── _InceptionBlock ───────────────────────────────────────────────────────────

class TestInceptionBlock:
    def test_output_shape(self):
        block = _InceptionBlock(d_model=16, kernels=(3, 7, 15, 31), dropout=0.0)
        x = torch.randn(B, 16, SEQ)
        assert block(x).shape == (B, 16, SEQ)

    def test_output_finite(self):
        block = _InceptionBlock(d_model=16, kernels=(3, 7, 15, 31), dropout=0.0)
        assert torch.isfinite(block(torch.randn(B, 16, SEQ))).all()

    def test_preserves_seq_length(self):
        for T in [8, 16, 32, 64]:
            block = _InceptionBlock(d_model=8, kernels=(3, 7), dropout=0.0)
            x = torch.randn(B, 8, T)
            assert block(x).shape[-1] == T

    def test_invalid_d_model_raises(self):
        with pytest.raises(AssertionError):
            _InceptionBlock(d_model=10, kernels=(3, 7, 15, 31), dropout=0.0)

    def test_gradient_flows(self):
        block = _InceptionBlock(d_model=16, kernels=(3, 7, 15, 31), dropout=0.0)
        x = torch.randn(B, 16, SEQ, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ── MultiscaleConvForecaster ──────────────────────────────────────────────────

class TestMultiscaleConvForecaster:
    def _model(self, **kw):
        defaults = dict(seq_len=SEQ, pred_len=PRED, enc_in=C,
                        d_model=16, n_layers=2,
                        kernels=(3, 7, 15, 31), dropout=0.0)
        defaults.update(kw)
        return MultiscaleConvForecaster(**defaults)

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

    def test_single_layer(self):
        m = self._model(n_layers=1)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_deeper_model(self):
        m = self._model(n_layers=4)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_larger_pred_len(self):
        m = self._model(pred_len=96)
        assert m(torch.randn(B, SEQ, C)).shape == (B, 96, C)

    def test_custom_kernels_two(self):
        m = self._model(d_model=8, kernels=(3, 7))
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_invalid_d_model_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            MultiscaleConvForecaster(seq_len=SEQ, pred_len=PRED, enc_in=C,
                                     d_model=10, kernels=(3, 7, 15, 31))

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

    def test_parallel_branches_exist(self):
        """Each inception block must have as many branches as kernels."""
        m = self._model(kernels=(3, 7, 15, 31))
        for block in m.blocks:
            assert len(block.branches) == 4

    def test_different_kernel_sizes(self):
        """Branches must have different kernel sizes."""
        m = self._model(kernels=(3, 7, 15, 31))
        block = m.blocks[0]
        ks = sorted(b.kernel_size[0] for b in block.branches)
        assert ks == [3, 7, 15, 31]

    def test_importable_from_package(self):
        from torch_timeseries.model import MultiscaleConvForecaster as MSC
        assert MSC is MultiscaleConvForecaster

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
