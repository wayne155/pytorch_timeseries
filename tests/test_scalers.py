"""Tests for torch_timeseries.scaler (MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler)."""
import numpy as np
import pytest
import torch

from torch_timeseries.scaler import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

RNG = np.random.default_rng(42)
DATA_NP = RNG.standard_normal((200, 7)).astype(np.float32)
# Add outliers to test robustness
DATA_NP[0] += 50
DATA_T = torch.from_numpy(DATA_NP)


# ── helpers ───────────────────────────────────────────────────────────────────

def _check_roundtrip(scaler, data):
    scaler.fit(data)
    t = scaler.transform(data)
    back = scaler.inverse_transform(t)
    if isinstance(data, np.ndarray):
        assert np.allclose(back, data, atol=1e-5), "numpy inverse_transform failed"
    else:
        assert torch.allclose(back, data, atol=1e-5), "tensor inverse_transform failed"


# ── MinMaxScaler ──────────────────────────────────────────────────────────────

class TestMinMaxScaler:
    def test_range_01_numpy(self):
        mm = MinMaxScaler()
        mm.fit(DATA_NP)
        t = mm.transform(DATA_NP)
        assert np.all(t >= -1e-6)
        assert np.all(t <= 1 + 1e-6)

    def test_min_is_0_max_is_1(self):
        mm = MinMaxScaler()
        mm.fit(DATA_NP)
        t = mm.transform(DATA_NP)
        assert np.allclose(t.min(axis=0), 0, atol=1e-5)
        assert np.allclose(t.max(axis=0), 1, atol=1e-5)

    def test_custom_feature_range(self):
        mm = MinMaxScaler(feature_range=(-1, 1))
        mm.fit(DATA_NP)
        t = mm.transform(DATA_NP)
        assert np.all(t >= -1 - 1e-6)
        assert np.all(t <= 1 + 1e-6)

    def test_roundtrip_numpy(self):
        _check_roundtrip(MinMaxScaler(), DATA_NP)

    def test_roundtrip_tensor(self):
        _check_roundtrip(MinMaxScaler(), DATA_T)

    def test_tensor_range(self):
        mm = MinMaxScaler()
        mm.fit(DATA_T)
        t = mm.transform(DATA_T)
        assert (t >= -1e-6).all()
        assert (t <= 1 + 1e-6).all()

    def test_invalid_feature_range(self):
        with pytest.raises(AssertionError):
            MinMaxScaler(feature_range=(1, 0))

    def test_constant_feature_no_nan(self):
        data = np.ones((50, 3), dtype=np.float32)
        mm = MinMaxScaler()
        mm.fit(data)
        t = mm.transform(data)
        assert not np.isnan(t).any()

    def test_unsupported_type(self):
        mm = MinMaxScaler()
        mm.fit(DATA_NP)
        with pytest.raises(ValueError):
            mm.transform([[1, 2], [3, 4]])


# ── RobustScaler ──────────────────────────────────────────────────────────────

class TestRobustScaler:
    def test_median_centered(self):
        rb = RobustScaler()
        # Simple 1-feature dataset
        data = np.arange(101, dtype=np.float32).reshape(-1, 1)
        rb.fit(data)
        t = rb.transform(data)
        assert abs(np.median(t)) < 1e-5

    def test_roundtrip_numpy(self):
        _check_roundtrip(RobustScaler(), DATA_NP)

    def test_roundtrip_tensor(self):
        _check_roundtrip(RobustScaler(), DATA_T)

    def test_custom_quantile_range(self):
        rb = RobustScaler(quantile_range=(10, 90))
        rb.fit(DATA_NP)
        t = rb.transform(DATA_NP)
        assert not np.isnan(t).any()

    def test_constant_feature_no_nan(self):
        data = np.ones((50, 3), dtype=np.float32)
        rb = RobustScaler()
        rb.fit(data)
        t = rb.transform(data)
        assert not np.isnan(t).any()

    def test_robust_to_outliers(self):
        # RobustScaler should give smaller scale than StandardScaler on outlier data
        std = StandardScaler()
        rb = RobustScaler()
        std.fit(DATA_NP)
        rb.fit(DATA_NP)
        t_std = std.transform(DATA_NP)
        t_rb = rb.transform(DATA_NP)
        # The outlier rows should have larger absolute values in RobustScaler output
        # because the scale is based on IQR, not std
        # verify scale parameters differ significantly
        assert not np.allclose(rb.scale_, std.std, rtol=0.1)

    def test_invalid_quantile_range(self):
        with pytest.raises(AssertionError):
            RobustScaler(quantile_range=(75, 25))


# ── StandardScaler (existing, regression test) ────────────────────────────────

class TestStandardScaler:
    def test_zero_mean(self):
        sc = StandardScaler()
        sc.fit(DATA_NP)
        t = sc.transform(DATA_NP)
        assert np.allclose(t.mean(axis=0), 0, atol=1e-5)

    def test_unit_std(self):
        sc = StandardScaler()
        sc.fit(DATA_NP)
        t = sc.transform(DATA_NP)
        assert np.allclose(t.std(axis=0), 1, atol=1e-4)

    def test_roundtrip(self):
        _check_roundtrip(StandardScaler(), DATA_NP)


# ── MaxAbsScaler (existing, regression test) ─────────────────────────────────

class TestMaxAbsScaler:
    def test_range_in_minus1_1(self):
        sc = MaxAbsScaler()
        sc.fit(DATA_NP)
        t = sc.transform(DATA_NP)
        assert np.all(np.abs(t) <= 1 + 1e-6)

    def test_roundtrip(self):
        _check_roundtrip(MaxAbsScaler(), DATA_NP)
