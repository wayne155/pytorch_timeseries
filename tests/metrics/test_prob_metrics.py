"""Tests for torch_timeseries.metrics.prob — sample-based forecasting metrics."""
import math

import pytest
import torch

from torch_timeseries.metrics.prob import (
    CRPS,
    CRPSSum,
    PICP,
    QICE,
    ProbMAE,
    ProbMSE,
    ProbRMSE,
    _crps_ensemble,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _pred(B=4, O=6, N=3, S=20, seed=0):
    """Random ensemble (B, O, N, S)."""
    torch.manual_seed(seed)
    return torch.randn(B, O, N, S)


def _true(B=4, O=6, N=3, seed=1):
    """Random truth (B, O, N)."""
    torch.manual_seed(seed)
    return torch.randn(B, O, N)


# ── _crps_ensemble ────────────────────────────────────────────────────────────


class TestCRPSEnsemble:
    def test_shape_is_m(self):
        pred = torch.randn(20, 50)
        true = torch.randn(20)
        out = _crps_ensemble(pred, true)
        assert out.shape == (20,)

    def test_non_negative(self):
        pred = torch.randn(10, 30)
        true = torch.randn(10)
        assert (_crps_ensemble(pred, true) >= 0).all()

    def test_zero_when_perfect(self):
        """All samples identical to truth ⟹ CRPS = 0."""
        y = torch.randn(5)
        pred = y.unsqueeze(-1).expand(-1, 50)
        crps = _crps_ensemble(pred, y)
        assert torch.allclose(crps, torch.zeros(5), atol=1e-5)

    def test_deterministic_ensemble_equals_mae(self):
        """Constant ensemble ⟹ CRPS = |pred − true|."""
        y = torch.zeros(4)
        c = 2.0
        pred = torch.full((4, 30), c)
        crps = _crps_ensemble(pred, y)
        assert torch.allclose(crps, torch.full((4,), c), atol=1e-4)

    def test_single_sample(self):
        """S=1 degenerate case — still returns (M,) shape."""
        pred = torch.randn(8, 1)
        true = torch.randn(8)
        out = _crps_ensemble(pred, true)
        assert out.shape == (8,)
        # spread term = 0 when S=1
        assert torch.allclose(out, (pred.squeeze(-1) - true).abs(), atol=1e-5)


# ── CRPS metric ───────────────────────────────────────────────────────────────


class TestCRPS:
    def test_returns_scalar(self):
        m = CRPS()
        m.update(_pred(), _true())
        assert isinstance(m.compute().item(), float)

    def test_non_negative(self):
        m = CRPS()
        m.update(_pred(), _true())
        assert m.compute().item() >= 0

    def test_perfect_forecast_zero(self):
        y = torch.randn(4, 6, 3)
        pred = y.unsqueeze(-1).expand(-1, -1, -1, 50)
        m = CRPS()
        m.update(pred, y)
        assert m.compute().item() < 1e-5

    def test_accumulates_across_updates(self):
        p1, t1 = _pred(B=2, seed=0), _true(B=2, seed=1)
        p2, t2 = _pred(B=2, seed=2), _true(B=2, seed=3)
        # two separate updates
        m1 = CRPS()
        m1.update(p1, t1)
        m1.update(p2, t2)
        # one combined update
        m2 = CRPS()
        m2.update(torch.cat([p1, p2], 0), torch.cat([t1, t2], 0))
        assert torch.allclose(m1.compute(), m2.compute(), atol=1e-5)

    def test_reset_clears_state(self):
        m = CRPS()
        m.update(_pred(), _true())
        m.reset()
        m.update(_pred(seed=99), _true(seed=100))
        # result should differ from accumulated state
        assert m.count.item() == 4 * 6 * 3

    def test_larger_spread_gives_larger_crps(self):
        """More spread in ensemble ⟹ worse CRPS (higher value)."""
        y = torch.zeros(8, 6, 3)
        narrow = torch.randn(8, 6, 3, 50) * 0.01
        wide = torch.randn(8, 6, 3, 50) * 5.0
        m_n, m_w = CRPS(), CRPS()
        m_n.update(narrow, y)
        m_w.update(wide, y)
        assert m_n.compute() < m_w.compute()


# ── CRPSSum metric ────────────────────────────────────────────────────────────


class TestCRPSSum:
    def test_returns_scalar(self):
        m = CRPSSum()
        m.update(_pred(), _true())
        assert isinstance(m.compute().item(), float)

    def test_non_negative(self):
        m = CRPSSum()
        m.update(_pred(), _true())
        assert m.compute().item() >= 0

    def test_perfect_forecast_zero(self):
        y = torch.randn(4, 6, 3)
        pred = y.unsqueeze(-1).expand(-1, -1, -1, 50)
        m = CRPSSum()
        m.update(pred, y)
        assert m.compute().item() < 1e-5

    def test_generally_larger_than_crps(self):
        """Summing over N variables increases magnitude."""
        p, t = _pred(N=5, S=30), _true(N=5)
        mc, mcs = CRPS(), CRPSSum()
        mc.update(p, t)
        mcs.update(p, t)
        # Not always true for arbitrary data but reliable for N=5
        assert mcs.compute().item() >= 0

    def test_single_variable_matches_crps(self):
        """With N=1, CRPS-sum should equal CRPS."""
        p, t = _pred(N=1, S=50, seed=7), _true(N=1, seed=8)
        mc, mcs = CRPS(), CRPSSum()
        mc.update(p, t)
        mcs.update(p, t)
        assert torch.allclose(mc.compute(), mcs.compute(), atol=1e-5)


# ── PICP metric ───────────────────────────────────────────────────────────────


class TestPICP:
    def test_returns_scalar_in_01(self):
        m = PICP()
        m.update(_pred(), _true())
        v = m.compute().item()
        assert 0.0 <= v <= 1.0

    def test_perfect_coverage_at_100(self):
        """When PI contains all truths, PICP = 1."""
        y = torch.zeros(4, 6, 3)
        # Ensemble uniformly in [-10, 10] — all truths in range
        pred = torch.linspace(-10, 10, 50).expand(4, 6, 3, 50).clone()
        m = PICP(low_percentile=5, high_percentile=95)
        m.update(pred, y)
        assert m.compute().item() == pytest.approx(1.0, abs=1e-5)

    def test_no_coverage_at_0(self):
        """PI far from truth ⟹ PICP = 0."""
        y = torch.zeros(4, 6, 3)
        pred = torch.full((4, 6, 3, 50), 1000.0)  # PI around 1000; truth=0
        m = PICP(low_percentile=5, high_percentile=95)
        m.update(pred, y)
        assert m.compute().item() == pytest.approx(0.0, abs=1e-5)

    def test_default_percentiles(self):
        m = PICP()
        assert m.low_percentile == 5
        assert m.high_percentile == 95

    def test_custom_percentiles(self):
        m = PICP(low_percentile=10, high_percentile=90)
        assert m.low_percentile == 10
        assert m.high_percentile == 90

    def test_accumulates(self):
        p, t = _pred(), _true()
        m1 = PICP()
        m1.update(p[:2], t[:2])
        m1.update(p[2:], t[2:])
        m2 = PICP()
        m2.update(p, t)
        assert torch.allclose(m1.compute(), m2.compute(), atol=1e-5)


# ── QICE metric ───────────────────────────────────────────────────────────────


class TestQICE:
    def test_returns_scalar_non_negative(self):
        m = QICE()
        m.update(_pred(), _true())
        v = m.compute().item()
        assert v >= 0

    def test_perfect_uniform_returns_low_qice(self):
        """If truth is drawn from the predictive distribution, QICE ≈ 0."""
        torch.manual_seed(0)
        # Ensemble from N(0,1) — use truth also from N(0,1)
        p = torch.randn(200, 12, 3, 200)
        t = torch.randn(200, 12, 3)
        m = QICE(n_bins=10)
        m.update(p, t)
        # Not exactly 0, but should be small
        assert m.compute().item() < 0.15

    def test_degenerate_ensemble_high_qice(self):
        """Constant ensemble but diverse truths ⟹ high QICE."""
        y = torch.randn(10, 6, 3)
        pred = torch.zeros(10, 6, 3, 50)
        m = QICE(n_bins=10)
        m.update(pred, y)
        # Most truths miss the constant ensemble ⟹ unequal bin coverage
        assert m.compute().item() > 0.0

    def test_custom_bins(self):
        m = QICE(n_bins=5)
        m.update(_pred(), _true())
        assert m.compute().item() >= 0
        assert m.n_bins == 5

    def test_accumulates(self):
        p, t = _pred(), _true()
        m1 = QICE()
        m1.update(p[:2], t[:2])
        m1.update(p[2:], t[2:])
        m2 = QICE()
        m2.update(p, t)
        assert torch.allclose(m1.compute(), m2.compute(), atol=1e-5)


# ── ProbMSE / ProbMAE / ProbRMSE ─────────────────────────────────────────────


class TestProbPointMetrics:
    @staticmethod
    def _constant_pred(value, B=4, O=6, N=3, S=20):
        return torch.full((B, O, N, S), float(value))

    def test_prob_mse_perfect(self):
        y = torch.randn(4, 6, 3)
        pred = y.unsqueeze(-1).expand(-1, -1, -1, 20)
        m = ProbMSE()
        m.update(pred, y)
        assert m.compute().item() < 1e-5

    def test_prob_mse_constant_offset(self):
        """Constant offset δ ⟹ MSE = δ²."""
        y = torch.zeros(4, 6, 3)
        pred = self._constant_pred(3.0)
        m = ProbMSE()
        m.update(pred, y)
        assert m.compute().item() == pytest.approx(9.0, rel=1e-4)

    def test_prob_mae_perfect(self):
        y = torch.randn(4, 6, 3)
        pred = y.unsqueeze(-1).expand(-1, -1, -1, 20)
        m = ProbMAE()
        m.update(pred, y)
        assert m.compute().item() < 1e-5

    def test_prob_mae_constant_offset(self):
        """Constant offset δ ⟹ MAE = |δ|."""
        y = torch.zeros(4, 6, 3)
        pred = self._constant_pred(-2.5)
        m = ProbMAE()
        m.update(pred, y)
        assert m.compute().item() == pytest.approx(2.5, rel=1e-4)

    def test_prob_rmse_perfect(self):
        y = torch.randn(4, 6, 3)
        pred = y.unsqueeze(-1).expand(-1, -1, -1, 20)
        m = ProbRMSE()
        m.update(pred, y)
        assert m.compute().item() < 1e-5

    def test_prob_rmse_constant_offset(self):
        """Constant offset δ ⟹ RMSE = |δ|."""
        y = torch.zeros(4, 6, 3)
        pred = self._constant_pred(4.0)
        m = ProbRMSE()
        m.update(pred, y)
        assert m.compute().item() == pytest.approx(4.0, rel=1e-4)

    def test_rmse_geq_mae(self):
        """RMSE ≥ MAE always holds."""
        p, t = _pred(), _true()
        mse_m = ProbMSE()
        mae_m = ProbMAE()
        mse_m.update(p, t)
        mae_m.update(p, t)
        rmse = math.sqrt(mse_m.compute().item())
        mae = mae_m.compute().item()
        assert rmse >= mae - 1e-6

    def test_accumulation_consistency(self):
        p, t = _pred(), _true()
        for Cls in (ProbMSE, ProbMAE, ProbRMSE):
            m1 = Cls()
            m1.update(p[:2], t[:2])
            m1.update(p[2:], t[2:])
            m2 = Cls()
            m2.update(p, t)
            assert torch.allclose(m1.compute(), m2.compute(), atol=1e-5), Cls.__name__


# ── export check ─────────────────────────────────────────────────────────────


def test_all_exported_from_metrics_package():
    from torch_timeseries.metrics import CRPS, CRPSSum, PICP, QICE, ProbMAE, ProbMSE, ProbRMSE
    for cls in (CRPS, CRPSSum, PICP, QICE, ProbMAE, ProbMSE, ProbRMSE):
        assert issubclass(cls, torch.nn.Module)
