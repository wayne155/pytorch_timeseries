"""Tests for torch_timeseries.metrics.point (SMAPE, MASE, QuantileLoss)."""
import math
import pytest
import torch
from torchmetrics import MetricCollection

from torch_timeseries.metrics import SMAPE, MASE, QuantileLoss, naive_seasonal_mae


# ── SMAPE ─────────────────────────────────────────────────────────────────────

def test_smape_zero_error():
    m = SMAPE()
    x = torch.tensor([1.0, 2.0, 3.0])
    m.update(x, x)
    assert math.isclose(m.compute().item(), 0.0, abs_tol=1e-6)


def test_smape_known_value():
    # pred=2, target=1 → 200 * |2-1| / (|2|+|1|) = 200/3 ≈ 66.67
    m = SMAPE()
    m.update(torch.tensor([2.0]), torch.tensor([1.0]))
    expected = 200 / 3
    assert math.isclose(m.compute().item(), expected, rel_tol=1e-5)


def test_smape_reset():
    m = SMAPE()
    m.update(torch.tensor([2.0]), torch.tensor([1.0]))
    m.reset()
    m.update(torch.tensor([1.0]), torch.tensor([1.0]))
    assert math.isclose(m.compute().item(), 0.0, abs_tol=1e-6)


def test_smape_batch_accumulation():
    m = SMAPE()
    x = torch.ones(10)
    y = torch.ones(10) * 2
    # batched
    for i in range(5):
        m.update(y[i * 2: (i + 1) * 2], x[i * 2: (i + 1) * 2])
    val_batched = m.compute().item()

    m2 = SMAPE()
    m2.update(y, x)
    val_single = m2.compute().item()
    assert math.isclose(val_batched, val_single, rel_tol=1e-5)


def test_smape_bounds():
    # SMAPE is in [0, 200]
    pred = torch.randn(100)
    target = torch.randn(100)
    m = SMAPE()
    m.update(pred, target)
    val = m.compute().item()
    assert 0.0 <= val <= 200.0


# ── QuantileLoss ──────────────────────────────────────────────────────────────

def test_quantile_loss_median_equals_half_mae():
    # At q=0.5, QL = 0.5 * MAE
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 3.5])
    mae = (pred - target).abs().mean().item()
    q = QuantileLoss(quantile=0.5)
    q.update(pred, target)
    assert math.isclose(q.compute().item(), mae / 2, rel_tol=1e-5)


def test_quantile_loss_zero_error():
    q = QuantileLoss(quantile=0.9)
    x = torch.randn(20)
    q.update(x, x)
    assert math.isclose(q.compute().item(), 0.0, abs_tol=1e-6)


def test_quantile_loss_asymmetry():
    # Over-forecast should be penalised less at high quantiles
    target = torch.zeros(100)
    over = torch.ones(100)      # pred > target
    under = -torch.ones(100)    # pred < target
    q_high = QuantileLoss(quantile=0.9)
    q_low  = QuantileLoss(quantile=0.1)
    q_high.update(over, target)
    q_low.update(over, target)
    assert q_high.compute().item() < q_low.compute().item()


def test_quantile_loss_invalid_quantile():
    with pytest.raises(AssertionError):
        QuantileLoss(quantile=0.0)
    with pytest.raises(AssertionError):
        QuantileLoss(quantile=1.0)


# ── MASE ──────────────────────────────────────────────────────────────────────

def test_mase_perfect_forecast():
    naive_mae = 1.0
    m = MASE(naive_mae=naive_mae)
    x = torch.tensor([1.0, 2.0, 3.0])
    m.update(x, x)
    assert math.isclose(m.compute().item(), 0.0, abs_tol=1e-6)


def test_mase_known_value():
    # MAE = 0.1, naive_mae = 1.0 → MASE = 0.1
    pred = torch.ones(10) * 1.1
    target = torch.ones(10)
    m = MASE(naive_mae=1.0)
    m.update(pred, target)
    assert math.isclose(m.compute().item(), 0.1, rel_tol=1e-5)


def test_mase_requires_naive_mae():
    m = MASE()
    m.update(torch.ones(5), torch.ones(5))
    with pytest.raises(RuntimeError, match="set_naive_mae"):
        m.compute()


def test_mase_set_naive_mae():
    m = MASE()
    m.set_naive_mae(2.0)
    pred = torch.ones(4) * 1.2
    target = torch.ones(4)
    m.update(pred, target)
    # MAE = 0.2, scale = 2.0 → 0.1
    assert math.isclose(m.compute().item(), 0.1, rel_tol=1e-5)


# ── naive_seasonal_mae ────────────────────────────────────────────────────────

def test_naive_seasonal_mae_rw():
    # constant series → MAE of seasonal naive = 0
    y = torch.ones(50)
    assert naive_seasonal_mae(y, seasonality=1) == 0.0


def test_naive_seasonal_mae_linear():
    # y = 0,1,2,...,99  →  y[t] - y[t-1] = 1 always
    y = torch.arange(100).float()
    assert math.isclose(naive_seasonal_mae(y, seasonality=1), 1.0, rel_tol=1e-5)


def test_naive_seasonal_mae_2d():
    # (T, C) input
    y = torch.arange(100).float().unsqueeze(-1).expand(-1, 3)
    assert math.isclose(naive_seasonal_mae(y, seasonality=1), 1.0, rel_tol=1e-5)


def test_naive_seasonal_mae_too_short():
    with pytest.raises(ValueError):
        naive_seasonal_mae(torch.ones(3), seasonality=5)


# ── MetricCollection integration ──────────────────────────────────────────────

def test_metric_collection_compatibility():
    mc = MetricCollection({
        "smape": SMAPE(),
        "q50": QuantileLoss(quantile=0.5),
        "mase": MASE(naive_mae=1.0),
    })
    pred = torch.randn(32)
    target = torch.randn(32)
    mc.update(pred, target)
    results = mc.compute()
    assert set(results.keys()) == {"smape", "q50", "mase"}
    for v in results.values():
        assert not torch.isnan(v)
