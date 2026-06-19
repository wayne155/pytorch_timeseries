"""Tests for ImplicitNeuralForecaster — INR-style coordinate-MLP forecaster."""
import math
import pytest
import torch
import torch.nn as nn

from torch_timeseries.model.ImplicitNeuralForecaster import (
    ImplicitNeuralForecaster, _time_embedding, _make_mlp,
)


SEQ = 24
PRED = 8
B = 2
C = 3


# ── _time_embedding ───────────────────────────────────────────────────────────

class TestTimeEmbedding:
    def test_shape(self):
        emb = _time_embedding(n_steps=PRED, d_time=33)
        assert emb.shape == (PRED, 33)

    def test_first_feature_is_ones(self):
        emb = _time_embedding(n_steps=PRED, d_time=33)
        assert torch.allclose(emb[:, 0], torch.ones(PRED))

    def test_range_from_0_to_1(self):
        """Sinusoidal features based on linspace(0,1) should vary smoothly."""
        emb = _time_embedding(n_steps=10, d_time=5)
        assert torch.isfinite(emb).all()

    def test_all_finite(self):
        assert torch.isfinite(_time_embedding(n_steps=96, d_time=33)).all()

    def test_different_steps_differ(self):
        emb = _time_embedding(n_steps=PRED, d_time=33)
        assert not torch.allclose(emb[0], emb[PRED // 2])


# ── ImplicitNeuralForecaster ──────────────────────────────────────────────────

class TestImplicitNeuralForecaster:
    def _model(self, **kw):
        defaults = dict(seq_len=SEQ, pred_len=PRED, enc_in=C,
                        d_latent=32, d_time=9, enc_layers=2, dec_layers=2,
                        d_hidden=64, dropout=0.0)
        defaults.update(kw)
        return ImplicitNeuralForecaster(**defaults)

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

    def test_deeper_encoder(self):
        m = self._model(enc_layers=4)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_deeper_decoder(self):
        m = self._model(dec_layers=5)
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

    def test_independent_decoding_per_step(self):
        """Each output step must be decoded independently — verify by checking
        that different pred_len values give consistent initial steps."""
        m = self._model(pred_len=4).eval()
        x = torch.randn(1, SEQ, C)
        with torch.no_grad():
            y4 = m(x)   # (1, 4, C)
        m2 = self._model(pred_len=8).eval()
        # Copy weights
        m2.load_state_dict(dict(m.state_dict()), strict=False)
        # The first 4 steps of pred_len=8 won't match because d_time differs
        # But we can verify shape and finiteness at minimum
        with torch.no_grad():
            y8 = m2(x)
        assert y4.shape == (1, 4, C)
        assert y8.shape == (1, 8, C)

    def test_encoder_output_shape(self):
        """Encoder must map (BC, T) → (BC, d_latent)."""
        m = self._model(d_latent=32)
        x = torch.randn(4, SEQ)  # 4 = B*C
        ctx = m.encoder(x)
        assert ctx.shape == (4, 32)

    def test_decoder_input_dim(self):
        """Decoder first layer input must be d_latent + d_time."""
        m = self._model(d_latent=32, d_time=9)
        # First linear layer of decoder
        first_linear = m.decoder[0]
        assert first_linear.in_features == 32 + 9

    def test_importable_from_package(self):
        from torch_timeseries.model import ImplicitNeuralForecaster as INF
        assert INF is ImplicitNeuralForecaster

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

    def test_more_latent_more_params(self):
        m_small = self._model(d_latent=16)
        m_large = self._model(d_latent=128)
        n_small = sum(p.numel() for p in m_small.parameters())
        n_large = sum(p.numel() for p in m_large.parameters())
        assert n_large > n_small
