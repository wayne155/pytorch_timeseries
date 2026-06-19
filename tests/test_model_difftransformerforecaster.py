"""Tests for DiffTransformerForecaster — differential attention (softmax1 - λ·softmax2)."""
import pytest
import torch

from torch_timeseries.model.DiffTransformerForecaster import (
    DiffTransformerForecaster,
    _DiffAttnHead,
    _DiffTransformerBlock,
)

SEQ = 16
PRED = 8
B = 2
C = 3
D = 16   # d_model; 16 // 4 heads = 4, which is even, so d_sub = 2
H = 4


class TestDiffAttnHead:
    def test_output_shape(self):
        block = _DiffAttnHead(d_model=D, n_heads=H, dropout=0.0)
        assert block(torch.randn(B, SEQ, D)).shape == (B, SEQ, D)

    def test_output_finite(self):
        block = _DiffAttnHead(d_model=D, n_heads=H, dropout=0.0)
        assert torch.isfinite(block(torch.randn(B, SEQ, D))).all()

    def test_gradient_flows(self):
        block = _DiffAttnHead(d_model=D, n_heads=H, dropout=0.0)
        x = torch.randn(B, SEQ, D, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_invalid_odd_sub_head(self):
        """d_model // n_heads must be even."""
        with pytest.raises(AssertionError):
            _DiffAttnHead(d_model=12, n_heads=4, dropout=0.0)   # 12//4=3, odd

    def test_lambda_computation(self):
        """Lambda must be a finite scalar per head."""
        block = _DiffAttnHead(d_model=D, n_heads=H, dropout=0.0)
        lam = block._lambda()
        assert lam.shape == (H,)
        assert torch.isfinite(lam).all()

    def test_single_timestep(self):
        block = _DiffAttnHead(d_model=D, n_heads=H, dropout=0.0)
        assert block(torch.randn(B, 1, D)).shape == (B, 1, D)


class TestDiffTransformerBlock:
    def test_output_shape(self):
        block = _DiffTransformerBlock(d_model=D, n_heads=H, d_ffn=64, dropout=0.0)
        assert block(torch.randn(B, SEQ, D)).shape == (B, SEQ, D)

    def test_output_finite(self):
        block = _DiffTransformerBlock(d_model=D, n_heads=H, d_ffn=64, dropout=0.0)
        assert torch.isfinite(block(torch.randn(B, SEQ, D))).all()


class TestDiffTransformerForecaster:
    def _model(self, **kw):
        defaults = dict(
            seq_len=SEQ, pred_len=PRED, enc_in=C,
            d_model=D, n_heads=H, d_ffn=64, n_layers=2, dropout=0.0,
        )
        defaults.update(kw)
        return DiffTransformerForecaster(**defaults)

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
        from torch_timeseries.model import DiffTransformerForecaster as DT
        assert DT is DiffTransformerForecaster

    def test_batch_size_one(self):
        m = self._model()
        assert m(torch.randn(1, SEQ, C)).shape == (1, PRED, C)

    def test_larger_d_model(self):
        # d_model=32, n_heads=4 → d_head=8 (even) ✓
        m = self._model(d_model=32, n_heads=4, d_ffn=128)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_longer_seq(self):
        m = DiffTransformerForecaster(
            seq_len=96, pred_len=PRED, enc_in=C,
            d_model=D, n_heads=H, d_ffn=64
        )
        assert m(torch.randn(B, 96, C)).shape == (B, PRED, C)

    def test_lambda_trained(self):
        """λ parameters should receive gradients."""
        m = self._model(n_layers=1)
        m(torch.randn(B, SEQ, C)).sum().backward()
        for block in m.blocks:
            lq1 = block.attn.lambda_q1.grad
            assert lq1 is not None and torch.isfinite(lq1).all()
