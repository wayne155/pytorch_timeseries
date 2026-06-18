"""Tests for WarmupCosineScheduler and WarmupLinearScheduler."""
import math
import pytest
import torch
from torch.optim import SGD

from torch_timeseries.utils import WarmupCosineScheduler, WarmupLinearScheduler


def _make_optimizer(lr: float = 1e-3):
    p = torch.nn.Parameter(torch.zeros(1))
    return SGD([p], lr=lr)


def _collect_lrs(sched_cls, *, warmup_steps, total_steps, base_lr=1e-3, eta_min=0.0):
    opt = _make_optimizer(base_lr)
    sched = sched_cls(opt, warmup_steps=warmup_steps, total_steps=total_steps,
                      eta_min=eta_min)
    lrs = []
    for _ in range(total_steps):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    return lrs


# ── WarmupCosineScheduler ────────────────────────────────────────────────────

class TestWarmupCosine:
    def test_warmup_increases_monotonically(self):
        lrs = _collect_lrs(WarmupCosineScheduler, warmup_steps=10, total_steps=100)
        warmup_lrs = lrs[:10]
        for a, b in zip(warmup_lrs, warmup_lrs[1:]):
            assert a <= b

    def test_peak_equals_base_lr(self):
        lrs = _collect_lrs(WarmupCosineScheduler, warmup_steps=10, total_steps=100,
                           base_lr=1e-3)
        assert math.isclose(max(lrs), 1e-3, rel_tol=1e-4)

    def test_decay_after_warmup(self):
        lrs = _collect_lrs(WarmupCosineScheduler, warmup_steps=10, total_steps=100)
        assert lrs[10] >= lrs[50] >= lrs[-1]

    def test_end_lr_is_eta_min(self):
        lrs = _collect_lrs(WarmupCosineScheduler, warmup_steps=5, total_steps=50,
                           eta_min=1e-6)
        assert math.isclose(lrs[-1], 1e-6, rel_tol=1e-3)

    def test_no_warmup_starts_near_base_lr(self):
        # With warmup_steps=0 the first step already has progress=1/total, so LR
        # is just below the peak — allow 1% tolerance.
        lrs = _collect_lrs(WarmupCosineScheduler, warmup_steps=0, total_steps=50)
        assert math.isclose(lrs[0], 1e-3, rel_tol=0.02)

    def test_invalid_total_steps(self):
        opt = _make_optimizer()
        with pytest.raises(AssertionError):
            WarmupCosineScheduler(opt, warmup_steps=10, total_steps=5)

    def test_negative_warmup_steps(self):
        opt = _make_optimizer()
        with pytest.raises(AssertionError):
            WarmupCosineScheduler(opt, warmup_steps=-1, total_steps=10)

    def test_multiple_param_groups(self):
        p1 = torch.nn.Parameter(torch.zeros(1))
        p2 = torch.nn.Parameter(torch.zeros(1))
        opt = SGD([
            {"params": [p1], "lr": 1e-3},
            {"params": [p2], "lr": 1e-4},
        ])
        sched = WarmupCosineScheduler(opt, warmup_steps=5, total_steps=50)
        for _ in range(50):
            opt.step()
            sched.step()
        lr1 = opt.param_groups[0]["lr"]
        lr2 = opt.param_groups[1]["lr"]
        # Both should converge to eta_min (0)
        assert lr1 < 1e-6
        assert lr2 < 1e-7


# ── WarmupLinearScheduler ────────────────────────────────────────────────────

class TestWarmupLinear:
    def test_warmup_increases_monotonically(self):
        lrs = _collect_lrs(WarmupLinearScheduler, warmup_steps=10, total_steps=100)
        warmup_lrs = lrs[:10]
        for a, b in zip(warmup_lrs, warmup_lrs[1:]):
            assert a <= b

    def test_decay_is_linear(self):
        lrs = _collect_lrs(WarmupLinearScheduler, warmup_steps=0, total_steps=100,
                           base_lr=1.0, eta_min=0.0)
        # After warmup=0 the LR should decay linearly: lrs[i] ≈ (100-i)/100
        for i in range(0, 80, 10):
            expected = (100 - (i + 1)) / 100
            assert math.isclose(lrs[i], expected, rel_tol=1e-3), (
                f"step={i}: got {lrs[i]:.4f}, expected {expected:.4f}"
            )

    def test_end_lr_is_eta_min(self):
        lrs = _collect_lrs(WarmupLinearScheduler, warmup_steps=5, total_steps=50,
                           eta_min=1e-6)
        assert math.isclose(lrs[-1], 1e-6, rel_tol=1e-3)

    def test_invalid_total_steps(self):
        opt = _make_optimizer()
        with pytest.raises(AssertionError):
            WarmupLinearScheduler(opt, warmup_steps=10, total_steps=5)
