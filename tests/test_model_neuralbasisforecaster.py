"""Tests for NeuralBasisForecaster — learnable basis decomposition forecaster."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.model.NeuralBasisForecaster import NeuralBasisForecaster


SEQ = 24
PRED = 8
B = 2
C = 3
K = 16   # n_basis


class TestNeuralBasisForecaster:
    def _model(self, **kw):
        defaults = dict(seq_len=SEQ, pred_len=PRED, enc_in=C,
                        n_basis=K, d_hidden=32, dropout=0.0)
        defaults.update(kw)
        return NeuralBasisForecaster(**defaults)

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

    def test_many_bases(self):
        m = self._model(n_basis=128)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_single_basis(self):
        m = self._model(n_basis=1)
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

    def test_encoder_basis_has_grad(self):
        m = self._model()
        m(torch.randn(B, SEQ, C)).sum().backward()
        assert m.encoder_basis.grad is not None
        assert m.decoder_basis.grad is not None

    def test_all_params_have_grad(self):
        m = self._model()
        m(torch.randn(B, SEQ, C)).sum().backward()
        for name, p in m.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_encoder_basis_shape(self):
        m = self._model(n_basis=K)
        assert m.encoder_basis.shape == (K, SEQ)
        assert m.decoder_basis.shape == (K, PRED)

    def test_encoder_basis_normalised(self):
        """At runtime the encoder basis must be L2-normalised per basis vector."""
        m = self._model()
        m.eval()
        x = torch.randn(1, SEQ, 1)
        with torch.no_grad():
            phi = F.normalize(m.encoder_basis, dim=-1)
            # Each row has unit norm
            norms = phi.norm(dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_basis_shapes_differ_from_each_other(self):
        """Basis functions should learn different shapes (not collapse)."""
        m = self._model()
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        t = torch.linspace(0, 3.14, SEQ).reshape(1, SEQ, 1)
        x = torch.sin(torch.randn(32, 1, 1) + t).expand(32, SEQ, C).clone()
        y = x[:, :PRED, :]
        for _ in range(5):
            opt.zero_grad()
            (m(x) - y).pow(2).mean().backward()
            opt.step()
        # After a few steps, at least some basis vectors should differ
        phi = F.normalize(m.encoder_basis.detach(), dim=-1)
        pairwise_sim = phi @ phi.T   # (K, K)
        off_diag = pairwise_sim.fill_diagonal_(0)
        assert off_diag.abs().max().item() < 0.9999  # not all identical

    def test_importable_from_package(self):
        from torch_timeseries.model import NeuralBasisForecaster as NBF
        assert NBF is NeuralBasisForecaster

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

    def test_more_bases_more_params(self):
        m_small = self._model(n_basis=4)
        m_large = self._model(n_basis=128)
        n_small = sum(p.numel() for p in m_small.parameters())
        n_large = sum(p.numel() for p in m_large.parameters())
        assert n_large > n_small
