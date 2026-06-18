"""Tests for Sine and Stocks generation benchmark datasets.

Sine is synthetic — no download required, fast to test.
Stocks requires a network download so tests are marked slow.
"""
import numpy as np
import pytest

from torch_timeseries.dataset.Sine import Sine

DATA_ROOT = "/tmp/torch_timeseries_test_data"


# ---------------------------------------------------------------------------
# Sine (synthetic — always fast)
# ---------------------------------------------------------------------------

class TestSineDataset:
    def _ds(self, n_points=1000, n_features=3, root="/tmp"):
        return Sine(n_points=n_points, n_features=n_features, root=root)

    def test_data_shape(self, tmp_path):
        ds = self._ds(n_points=500, n_features=4, root=str(tmp_path))
        assert ds.data.shape == (500, 4)

    def test_length_attribute(self, tmp_path):
        ds = self._ds(n_points=800, root=str(tmp_path))
        assert ds.length == 800

    def test_num_features_attribute(self, tmp_path):
        ds = self._ds(n_features=7, root=str(tmp_path))
        assert ds.num_features == 7

    def test_data_is_float32(self, tmp_path):
        ds = self._ds(root=str(tmp_path))
        assert ds.data.dtype == np.float32

    def test_data_range(self, tmp_path):
        ds = self._ds(n_points=2000, root=str(tmp_path))
        assert ds.data.min() >= -1.0
        assert ds.data.max() <= 1.0

    def test_data_no_nan(self, tmp_path):
        ds = self._ds(root=str(tmp_path))
        assert not np.isnan(ds.data).any()

    def test_data_reproducible(self, tmp_path):
        ds1 = self._ds(n_points=200, root=str(tmp_path))
        ds2 = self._ds(n_points=200, root=str(tmp_path))
        np.testing.assert_array_equal(ds1.data, ds2.data)

    def test_df_has_date_column(self, tmp_path):
        ds = self._ds(root=str(tmp_path))
        assert "date" in ds.df.columns

    def test_dates_dataframe_shape(self, tmp_path):
        ds = self._ds(n_points=300, root=str(tmp_path))
        assert ds.dates.shape == (300, 1)

    def test_default_n_points(self, tmp_path):
        ds = Sine(root=str(tmp_path))
        assert ds.length == 10_000

    def test_default_n_features(self, tmp_path):
        ds = Sine(root=str(tmp_path))
        assert ds.num_features == 5

    def test_single_feature(self, tmp_path):
        ds = self._ds(n_features=1, root=str(tmp_path))
        assert ds.data.shape[1] == 1

    def test_freq_is_h(self, tmp_path):
        ds = self._ds(root=str(tmp_path))
        assert ds.freq == "h"

    def test_name_is_sine(self, tmp_path):
        ds = self._ds(root=str(tmp_path))
        assert ds.name == "Sine"


# ---------------------------------------------------------------------------
# Stocks (requires network — marked slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestStocksDataset:
    def test_data_shape(self):
        from torch_timeseries.dataset.Stocks import Stocks
        ds = Stocks(root=DATA_ROOT)
        assert ds.data.ndim == 2
        assert ds.data.shape[1] == 6

    def test_num_features(self):
        from torch_timeseries.dataset.Stocks import Stocks
        ds = Stocks(root=DATA_ROOT)
        assert ds.num_features == 6

    def test_data_is_float32(self):
        from torch_timeseries.dataset.Stocks import Stocks
        ds = Stocks(root=DATA_ROOT)
        assert ds.data.dtype == np.float32

    def test_data_range_normalised(self):
        from torch_timeseries.dataset.Stocks import Stocks
        ds = Stocks(root=DATA_ROOT)
        assert ds.data.min() >= 0.0
        assert ds.data.max() <= 1.0

    def test_data_no_nan(self):
        from torch_timeseries.dataset.Stocks import Stocks
        ds = Stocks(root=DATA_ROOT)
        assert not np.isnan(ds.data).any()

    def test_length_attribute(self):
        from torch_timeseries.dataset.Stocks import Stocks
        ds = Stocks(root=DATA_ROOT)
        assert ds.length == len(ds.data)

    def test_freq_is_d(self):
        from torch_timeseries.dataset.Stocks import Stocks
        ds = Stocks(root=DATA_ROOT)
        assert ds.freq == "d"
