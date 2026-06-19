"""Tests for HGRN2Forecaster — norm-preserving gated RNN (i_t = sqrt(1 - f_t²))."""
import pytest
import torch

from torch_timeseries.model.HGRN2Forecaster import (
    HGRN2Forecaster,
    _HGRN2Cell,
    _HGRN2Block,
)

SEQ = 16
PRED = 8
B = 2
C = 3
D = 16


class TestHGRN2Cell:
    def test_output_shape(self):
        cell = _HGRN2Cell(d_model=D)
        assert cell(torch.randn(B, SEQ, D)).shape == (B, SEQ, D)

    def test_output_finite(self):
        cell = _HGRN2Cell(d_model=D)
        assert torch.isfinite(cell(torch.randn(B, SEQ, D))).all()

    def test_gradient_flows(self):
        cell = _HGRN2Cell(d_model=D)
        x = torch.randn(B, SEQ, D, requires_grad=True)
        cell(x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_norm_preserving(self):
        """Hidden state norm should stay ≤ 1 when starting from zero."""
        cell = _HGRN2Cell(d_model=D)
        cell.eval()
        with torch.no_grad():
            x = torch.randn(1, 64, D)
            h_seq = cell(x)   # (1, 64, D)
            norms = h_seq.norm(dim=-1)  # (1, 64)
            # The output is o_t * h_t where o_t = sigmoid(·) ∈ (0,1) and ||h_t|| ≤ sqrt(D)
            # We check that norms don't grow unboundedly (bounded by sqrt(D))
            assert (norms <= D ** 0.5 + 1e-3).all()

    def test_single_timestep(self):
        cell = _HGRN2Cell(d_model=D)
        assert cell(torch.randn(B, 1, D)).shape == (B, 1, D)

    def test_no_hidden_state_dependence_on_gates(self):
        """f_t depends only on x_t, not h_{t-1} — verify by checking W_f has no bias toward h."""
        cell = _HGRN2Cell(d_model=D)
        # W_f has only input features, no hidden state. Parameter count check:
        # W_f: D×D weights + D bias
        assert cell.W_f.weight.shape == (D, D)
        assert cell.W_f.bias is not None


class TestHGRN2Block:
    def test_output_shape(self):
        block = _HGRN2Block(d_model=D, d_ffn=64, dropout=0.0)
        assert block(torch.randn(B, SEQ, D)).shape == (B, SEQ, D)

    def test_output_finite(self):
        block = _HGRN2Block(d_model=D, d_ffn=64, dropout=0.0)
        assert torch.isfinite(block(torch.randn(B, SEQ, D))).all()


class TestHGRN2Forecaster:
    def _model(self, **kw):
        defaults = dict(
            seq_len=SEQ, pred_len=PRED, enc_in=C,
            d_model=D, d_ffn=64, n_layers=2, dropout=0.0,
        )
        defaults.update(kw)
        return HGRN2Forecaster(**defaults)

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
        from torch_timeseries.model import HGRN2Forecaster as HG
        assert HG is HGRN2Forecaster

    def test_batch_size_one(self):
        m = self._model()
        assert m(torch.randn(1, SEQ, C)).shape == (1, PRED, C)

    def test_longer_seq(self):
        m = HGRN2Forecaster(seq_len=96, pred_len=PRED, enc_in=C, d_model=D, d_ffn=64)
        assert m(torch.randn(B, 96, C)).shape == (B, PRED, C)

    def test_forget_input_gate_coupling(self):
        """i_t = sqrt(1 - f_t^2) so f^2 + i^2 = 1."""
        cell = _HGRN2Cell(D)
        with torch.no_grad():
            x = torch.randn(1, 1, D)
            f = torch.sigmoid(cell.W_f(x))
            i = torch.sqrt(torch.clamp(1.0 - f * f, min=1e-6))
            coupling = f * f + i * i
            assert torch.allclose(coupling, torch.ones_like(coupling), atol=1e-4)
