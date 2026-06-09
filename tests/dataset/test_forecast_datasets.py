"""Integration tests: verify that forecast datasets can be downloaded and loaded.

Run with:  python3.13 -m pytest tests/dataset/test_forecast_datasets.py -m slow -v

Downloads are cached at DATA_ROOT, so subsequent runs skip the network step.
"""
import numpy as np
import pytest

# Shared download cache — persists between test runs so files aren't re-fetched.
DATA_ROOT = "/tmp/torch_timeseries_test_data"

pytestmark = pytest.mark.slow  # skip in fast CI with: pytest -m "not slow"


# ── dataset registry ─────────────────────────────────────────────────────────
#   (class_name, expected_length, expected_num_features)
FORECAST_DATASETS = [
    ("ETTh1",       17420,  7),
    ("ETTh2",       17420,  7),
    ("ETTm1",       69680,  7),
    ("ETTm2",       69680,  7),
    ("Weather",     52696, 21),
    ("Traffic",     17544, 862),
    ("Electricity", 26304, 321),
    ("ILI",           966,  7),
    ("ExchangeRate", 7588,  8),
]


def _load(name):
    from torch_timeseries.dataset import (
        ETTh1, ETTh2, ETTm1, ETTm2, Weather,
        Traffic, Electricity, ILI, ExchangeRate,
    )
    cls = {
        "ETTh1": ETTh1, "ETTh2": ETTh2, "ETTm1": ETTm1, "ETTm2": ETTm2,
        "Weather": Weather, "Traffic": Traffic, "Electricity": Electricity,
        "ILI": ILI, "ExchangeRate": ExchangeRate,
    }[name]
    return cls(root=DATA_ROOT)


# ── parametrized smoke tests ──────────────────────────────────────────────────

@pytest.mark.parametrize("name, length, n_feat", FORECAST_DATASETS)
def test_data_shape(name, length, n_feat):
    ds = _load(name)
    assert ds.data.shape == (length, n_feat), (
        f"{name}: expected ({length}, {n_feat}), got {ds.data.shape}"
    )


@pytest.mark.parametrize("name, length, n_feat", FORECAST_DATASETS)
def test_length_attribute_matches_data(name, length, n_feat):
    ds = _load(name)
    assert len(ds.data) == ds.length


@pytest.mark.parametrize("name, length, n_feat", FORECAST_DATASETS)
def test_num_features_attribute_matches_data(name, length, n_feat):
    ds = _load(name)
    assert ds.data.shape[1] == ds.num_features


@pytest.mark.parametrize("name, length, n_feat", FORECAST_DATASETS)
def test_dates_present(name, length, n_feat):
    ds = _load(name)
    assert ds.dates is not None
    assert "date" in ds.dates.columns
    assert len(ds.dates) == length


@pytest.mark.parametrize("name, length, n_feat", FORECAST_DATASETS)
def test_dates_are_datetime(name, length, n_feat):
    import pandas as pd
    ds = _load(name)
    assert pd.api.types.is_datetime64_any_dtype(ds.dates["date"])


@pytest.mark.parametrize("name, length, n_feat", FORECAST_DATASETS)
def test_data_is_numeric(name, length, n_feat):
    ds = _load(name)
    assert ds.data.dtype.kind in ("f", "i", "u"), (
        f"{name}: unexpected dtype {ds.data.dtype}"
    )


@pytest.mark.parametrize("name, length, n_feat", FORECAST_DATASETS)
def test_df_has_date_and_feature_columns(name, length, n_feat):
    ds = _load(name)
    assert "date" in ds.df.columns
    assert ds.df.shape == (length, n_feat + 1)  # features + date


# ── per-dataset sanity checks ────────────────────────────────────────────────

def test_etth1_no_nan():
    ds = _load("ETTh1")
    assert not np.isnan(ds.data).any(), "ETTh1 contains NaN values"


def test_etth2_no_nan():
    ds = _load("ETTh2")
    assert not np.isnan(ds.data).any()


def test_ettm1_no_nan():
    ds = _load("ETTm1")
    assert not np.isnan(ds.data).any()


def test_ettm2_no_nan():
    ds = _load("ETTm2")
    assert not np.isnan(ds.data).any()


def test_weather_no_nan():
    ds = _load("Weather")
    assert not np.isnan(ds.data).any()


def test_traffic_values_in_range():
    """Traffic is road occupancy: values should be in [0, 1]."""
    ds = _load("Traffic")
    assert ds.data.min() >= 0.0
    assert ds.data.max() <= 1.0


def test_exchange_rate_positive():
    """Exchange rates must be strictly positive."""
    ds = _load("ExchangeRate")
    assert ds.data.min() > 0.0


def test_ili_no_nan():
    ds = _load("ILI")
    assert not np.isnan(ds.data).any()


def test_etth1_dates_monotone():
    """Dates must be strictly increasing."""
    import pandas as pd
    ds = _load("ETTh1")
    diffs = ds.dates["date"].diff().dropna()
    assert (diffs > pd.Timedelta(0)).all(), "ETTh1 dates are not monotonically increasing"


def test_weather_dates_monotone():
    # The upstream source file has one duplicate timestamp (2020-05-12 06:00),
    # so we check non-decreasing rather than strictly increasing.
    import pandas as pd
    ds = _load("Weather")
    diffs = ds.dates["date"].diff().dropna()
    assert (diffs >= pd.Timedelta(0)).all()


def test_traffic_dates_monotone():
    import pandas as pd
    ds = _load("Traffic")
    diffs = ds.dates["date"].diff().dropna()
    assert (diffs > pd.Timedelta(0)).all()


def test_etth1_freq():
    ds = _load("ETTh1")
    assert ds.freq in ("h", "H")


def test_ettm1_freq():
    ds = _load("ETTm1")
    assert ds.freq in ("t", "T", "min")
