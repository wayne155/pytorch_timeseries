"""Tests for RWKVForecaster — RWKV linear-attention-as-recurrence forecaster."""
import pytest
import torch

from torch_timeseries.model.RWKVForecaster import (
    RWKVForecaster, _TimeFirst, _ChannelMix, _RWKVBlock,
)

SEQ = 16
PRED = 8
B = 2
C = 3
D = 16


class TestTimeFirst:
    def test_output_shape(self):
        block = _TimeFirst(d_model=D)
        x = torch.randn(B, SEQ, D)
        out = block(x)
        assert out.shape == (B, SEQ, D)

    def test_output_finite(self):
        block = _TimeFirst(d_model=D)
        out = block(torch.randn(B, SEQ, D))
        assert torch.isfinite(out).all()

    def test_w_is_negative(self):
        """Decay w must be < 0 after softplus transform."""
        block = _TimeFirst(d_model=D)
        import torch.nn.functional as F
        w = -F.softplus(block.w_raw)
        assert (w < 0).all()


class TestChannelMix:
    def test_output_shape(self):
        mix = _ChannelMix(d_model=D, d_ffn=D * 4)
        assert mix(torch.randn(B, SEQ, D)).shape == (B, SEQ, D)

    def test_output_finite(self):
        mix = _ChannelMix(d_model=D, d_ffn=D * 4)
        assert torch.isfinite(mix(torch.randn(B, SEQ, D))).all()


class TestRWKVBlock:
    def test_output_shape(self):
        block = _RWKVBlock(d_model=D, d_ffn=D * 4)
        assert block(torch.randn(B, SEQ, D)).shape == (B, SEQ, D)

    def test_output_finite(self):
        block = _RWKVBlock(d_model=D, d_ffn=D * 4)
        assert torch.isfinite(block(torch.randn(B, SEQ, D))).all()


class TestRWKVForecaster:
    def _model(self, **kw):
        defaults = dict(seq_len=SEQ, pred_len=PRED, enc_in=C,
                        d_model=D, d_ffn=D * 4, n_layers=2, dropout=0.0)
        defaults.update(kw)
        return RWKVForecaster(**defaults)

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
        from torch_timeseries.model import RWKVForecaster as RWKV
        assert RWKV is RWKVForecaster

    def test_batch_size_one(self):
        m = self._model()
        assert m(torch.randn(1, SEQ, C)).shape == (1, PRED, C)

    def test_larger_d_model(self):
        m = self._model(d_model=64, d_ffn=256)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_longer_seq(self):
        m = RWKVForecaster(seq_len=48, pred_len=PRED, enc_in=C, d_model=D, d_ffn=D*4)
        assert m(torch.randn(B, 48, C)).shape == (B, PRED, C)
