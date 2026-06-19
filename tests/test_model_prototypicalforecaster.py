"""Tests for PrototypicalForecaster — prototype memory retrieval forecaster."""
import pytest
import torch
import torch.nn as nn

from torch_timeseries.model.PrototypicalForecaster import (
    PrototypicalForecaster, _QueryEncoder,
)


SEQ = 24
PRED = 8
B = 2
C = 3
K = 16   # n_proto


# ── _QueryEncoder ─────────────────────────────────────────────────────────────

class TestQueryEncoder:
    def test_output_shape(self):
        enc = _QueryEncoder(seq_len=SEQ, d_proto=32, query_dim=64, dropout=0.0)
        x = torch.randn(B, SEQ)
        assert enc(x).shape == (B, 32)

    def test_output_finite(self):
        enc = _QueryEncoder(seq_len=SEQ, d_proto=32, query_dim=64, dropout=0.0)
        assert torch.isfinite(enc(torch.randn(B, SEQ))).all()

    def test_layer_norm_output(self):
        """LayerNorm at the end should normalise per-vector."""
        enc = _QueryEncoder(seq_len=SEQ, d_proto=32, query_dim=64, dropout=0.0)
        q = enc(torch.randn(4, SEQ))
        # Mean ~ 0, variance ~ 1 per sample (within tolerance for small d)
        assert q.std(dim=-1).mean().item() < 2.0


# ── PrototypicalForecaster ────────────────────────────────────────────────────

class TestPrototypicalForecaster:
    def _model(self, **kw):
        defaults = dict(seq_len=SEQ, pred_len=PRED, enc_in=C,
                        n_proto=K, d_proto=32, query_dim=64, dropout=0.0)
        defaults.update(kw)
        return PrototypicalForecaster(**defaults)

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

    def test_more_prototypes(self):
        m = self._model(n_proto=128)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_fewer_prototypes(self):
        m = self._model(n_proto=1)
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

    def test_proto_keys_have_grad(self):
        m = self._model()
        m(torch.randn(B, SEQ, C)).sum().backward()
        assert m.proto_keys.grad is not None
        assert m.proto_vals.grad is not None

    def test_all_params_have_grad(self):
        m = self._model()
        m(torch.randn(B, SEQ, C)).sum().backward()
        for name, p in m.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_temperature_is_parameter(self):
        m = self._model()
        assert isinstance(m.log_temp, nn.Parameter)
        assert m.log_temp.shape == (1,)

    def test_proto_shapes(self):
        m = self._model(n_proto=K, d_proto=32)
        assert m.proto_keys.shape == (K, 32)
        assert m.proto_vals.shape == (K, PRED)

    def test_retrieval_uses_softmax(self):
        """The attention over prototypes must sum to 1."""
        import torch.nn.functional as F
        m = self._model()
        m.eval()
        x = torch.randn(1, SEQ, 1)
        with torch.no_grad():
            query = m.query_enc(x.squeeze(-1))  # (1, d_proto)
            scale = m._scale * m.log_temp.exp().clamp(min=0.1)
            sim = (query @ m.proto_keys.T) / scale
            attn = F.softmax(sim, dim=-1)
            assert torch.allclose(attn.sum(dim=-1), torch.ones(1), atol=1e-5)

    def test_importable_from_package(self):
        from torch_timeseries.model import PrototypicalForecaster as PF
        assert PF is PrototypicalForecaster

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

    def test_more_protos_more_params(self):
        m_small = self._model(n_proto=4)
        m_large = self._model(n_proto=64)
        n_small = sum(p.numel() for p in m_small.parameters())
        n_large = sum(p.numel() for p in m_large.parameters())
        assert n_large > n_small
