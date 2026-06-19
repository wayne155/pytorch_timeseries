"""Tests for LRUForecaster — Linear Recurrent Unit forecaster."""
import pytest
import torch
import torch.nn as nn

from torch_timeseries.model.LRUForecaster import LRUForecaster, _LRUCell, _LRUBlock


SEQ = 24
PRED = 8
B = 2
C = 3


# ── _LRUCell ──────────────────────────────────────────────────────────────────

class TestLRUCell:
    def test_output_shape(self):
        cell = _LRUCell(d_model=16, d_state=32, dropout=0.0)
        x = torch.randn(B, SEQ, 16)
        y = cell(x)
        assert y.shape == (B, SEQ, 16)

    def test_eigenvalues_stable(self):
        """All |Λ_i| must be strictly inside the unit disk."""
        cell = _LRUCell(d_model=8, d_state=16, dropout=0.0)
        lam_abs = torch.sigmoid(cell.nu)
        assert (lam_abs < 1.0).all()
        assert (lam_abs > 0.0).all()

    def test_output_finite(self):
        cell = _LRUCell(d_model=16, d_state=32, dropout=0.0)
        x = torch.randn(B, SEQ, 16)
        y = cell(x)
        assert torch.isfinite(y).all()

    def test_gradient_flows(self):
        cell = _LRUCell(d_model=8, d_state=8, dropout=0.0)
        x = torch.randn(2, 4, 8, requires_grad=True)
        y = cell(x)
        y.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_different_batch_sizes(self):
        cell = _LRUCell(d_model=16, d_state=16, dropout=0.0)
        for bs in [1, 4, 8]:
            y = cell(torch.randn(bs, SEQ, 16))
            assert y.shape == (bs, SEQ, 16)


# ── _LRUBlock ─────────────────────────────────────────────────────────────────

class TestLRUBlock:
    def test_output_shape(self):
        block = _LRUBlock(d_model=16, d_state=16, mlp_mult=2, dropout=0.0)
        x = torch.randn(B, SEQ, 16)
        y = block(x)
        assert y.shape == (B, SEQ, 16)

    def test_output_finite(self):
        block = _LRUBlock(d_model=16, d_state=16, mlp_mult=2, dropout=0.0)
        x = torch.randn(B, SEQ, 16)
        assert torch.isfinite(block(x)).all()


# ── LRUForecaster ─────────────────────────────────────────────────────────────

class TestLRUForecaster:
    def _model(self, **kw):
        defaults = dict(seq_len=SEQ, pred_len=PRED, enc_in=C, d_model=16,
                        d_state=16, n_layers=2, mlp_mult=2, dropout=0.0)
        defaults.update(kw)
        return LRUForecaster(**defaults)

    def test_output_shape(self):
        m = self._model()
        x = torch.randn(B, SEQ, C)
        y = m(x)
        assert y.shape == (B, PRED, C)

    def test_output_finite(self):
        m = self._model()
        x = torch.randn(B, SEQ, C)
        assert torch.isfinite(m(x)).all()

    def test_no_revin(self):
        m = self._model(revin=False)
        x = torch.randn(B, SEQ, C)
        assert m(x).shape == (B, PRED, C)

    def test_single_channel(self):
        m = self._model(enc_in=1)
        x = torch.randn(B, SEQ, 1)
        assert m(x).shape == (B, PRED, 1)

    def test_single_layer(self):
        m = self._model(n_layers=1)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_deeper_model(self):
        m = self._model(n_layers=4, d_model=32, d_state=32)
        assert m(torch.randn(B, SEQ, C)).shape == (B, PRED, C)

    def test_larger_pred_len(self):
        m = self._model(pred_len=96)
        assert m(torch.randn(B, SEQ, C)).shape == (B, 96, C)

    def test_many_channels(self):
        m = self._model(enc_in=21)
        assert m(torch.randn(B, SEQ, 21)).shape == (B, PRED, 21)

    def test_eval_mode_deterministic(self):
        m = self._model(dropout=0.1).eval()
        x = torch.randn(B, SEQ, C)
        with torch.no_grad():
            y1 = m(x)
            y2 = m(x)
        assert torch.allclose(y1, y2)

    def test_gradient_flows(self):
        m = self._model()
        x = torch.randn(B, SEQ, C, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_all_params_have_grad(self):
        m = self._model()
        x = torch.randn(B, SEQ, C)
        m(x).sum().backward()
        for name, p in m.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_complex_eigenvalue_structure(self):
        """LRU eigenvalues must be parameterised — nu and theta must exist."""
        m = self._model()
        lru_cell = m.blocks[0].lru
        assert hasattr(lru_cell, "nu")
        assert hasattr(lru_cell, "theta")
        assert isinstance(lru_cell.nu, nn.Parameter)
        assert isinstance(lru_cell.theta, nn.Parameter)

    def test_no_dense_state_matrix(self):
        """The cell must NOT have a weight matrix of shape (d_state, d_state)."""
        m = self._model(d_model=16, d_state=16)
        lru_cell = m.blocks[0].lru
        for name, p in lru_cell.named_parameters():
            if p.shape == (16, 16):
                # Only B/C projection matrices are square if d_model == d_state;
                # they are input/output projections, not state-to-state.
                # The state-to-state transition is the diagonal Λ, not a matrix.
                assert name in ("B_re", "B_im", "C_re", "C_im"), (
                    f"Unexpected (d_state, d_state) parameter: {name}"
                )

    def test_importable_from_package(self):
        from torch_timeseries.model import LRUForecaster as LRU
        assert LRU is LRUForecaster

    def test_training_step(self):
        m = self._model()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        x = torch.randn(4, SEQ, C)
        y = torch.randn(4, PRED, C)
        for _ in range(3):
            opt.zero_grad()
            loss = (m(x) - y).pow(2).mean()
            loss.backward()
            opt.step()
        assert torch.isfinite(loss)
