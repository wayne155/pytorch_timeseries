"""Tests for the high-level Forecaster fit/predict/score API."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from torch_timeseries.forecaster import Forecaster, compare, _WindowDataset, _EarlyStopping


# ── helpers ────────────────────────────────────────────────────────────────────

SEQ = 24
PRED = 8
N = 300
C = 3


def _rng_data(n=N, c=C, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, c)).astype(np.float32)


def _quick_fc(model="DLinear", **kw):
    return Forecaster(
        model, seq_len=SEQ, pred_len=PRED,
        epochs=2, batch_size=32, patience=5, verbose=False,
        **kw
    )


# ── WindowDataset ─────────────────────────────────────────────────────────────


class TestWindowDataset:
    def test_length(self):
        X = np.zeros((100, 3), dtype=np.float32)
        ds = _WindowDataset(X, seq_len=24, pred_len=8)
        assert len(ds) == 100 - 24 - 8 + 1

    def test_shapes(self):
        X = np.zeros((100, 3), dtype=np.float32)
        ds = _WindowDataset(X, seq_len=24, pred_len=8)
        x, y = ds[0]
        assert x.shape == (24, 3)
        assert y.shape == (8, 3)

    def test_last_window(self):
        X = np.zeros((100, 3), dtype=np.float32)
        ds = _WindowDataset(X, seq_len=24, pred_len=8)
        x, y = ds[len(ds) - 1]
        assert x.shape == (24, 3)
        assert y.shape == (8, 3)

    def test_too_short_gives_zero_len(self):
        X = np.zeros((10, 3), dtype=np.float32)
        ds = _WindowDataset(X, seq_len=24, pred_len=8)
        assert len(ds) == 0


# ── EarlyStopping ─────────────────────────────────────────────────────────────


class TestEarlyStopping:
    def test_stops_after_patience(self):
        model = nn.Linear(2, 2)
        es = _EarlyStopping(patience=3)
        # Improve once
        es(1.0, model)
        assert not es.stop
        # Stall for patience rounds
        es(2.0, model)
        es(2.0, model)
        es(2.0, model)
        assert es.stop

    def test_best_weights_captured(self):
        model = nn.Linear(2, 2)
        nn.init.constant_(model.weight, 0.5)
        es = _EarlyStopping(patience=2)
        es(1.0, model)
        # Change weights, no improvement
        nn.init.constant_(model.weight, 99.0)
        es(2.0, model)
        es.restore_best(model)
        assert (model.weight == 0.5).all()

    def test_counter_resets_on_improvement(self):
        model = nn.Linear(2, 2)
        es = _EarlyStopping(patience=3)
        es(2.0, model)
        es(1.9, model)
        assert es.counter == 0


# ── Forecaster construction ───────────────────────────────────────────────────


class TestForecasterConstruction:
    def test_repr_not_fitted(self):
        fc = _quick_fc()
        assert "not fitted" in repr(fc)

    def test_predict_raises_before_fit(self):
        fc = _quick_fc()
        with pytest.raises(RuntimeError, match="not fitted"):
            fc.predict(np.zeros((SEQ, C)))

    def test_score_raises_before_fit(self):
        fc = _quick_fc()
        with pytest.raises(RuntimeError, match="not fitted"):
            fc.score(np.zeros((100, C)))

    def test_fit_too_short_raises(self):
        fc = _quick_fc()
        with pytest.raises(ValueError, match="too short|timesteps"):
            fc.fit(np.zeros((5, C)))

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            Forecaster("NonExistentModel123", seq_len=SEQ, pred_len=PRED, epochs=1).fit(
                _rng_data()
            )


# ── Forecaster fit ────────────────────────────────────────────────────────────


class TestForecasterFit:
    def test_returns_self(self):
        fc = _quick_fc()
        result = fc.fit(_rng_data())
        assert result is fc

    def test_model_is_nn_module_after_fit(self):
        fc = _quick_fc().fit(_rng_data())
        assert isinstance(fc.model, nn.Module)

    def test_repr_fitted(self):
        fc = _quick_fc().fit(_rng_data())
        assert "fitted" in repr(fc)
        assert "not fitted" not in repr(fc)

    def test_n_parameters_positive(self):
        fc = _quick_fc().fit(_rng_data())
        assert fc.n_parameters > 0

    def test_scaler_fitted_when_normalize(self):
        fc = _quick_fc(normalize=True).fit(_rng_data())
        assert fc._scaler is not None
        assert fc._scaler.mean is not None

    def test_no_scaler_when_no_normalize(self):
        fc = _quick_fc(normalize=False).fit(_rng_data())
        assert fc._scaler is None

    def test_single_channel_accepted(self):
        X = _rng_data(c=1)
        fc = _quick_fc().fit(X)
        assert fc._enc_in == 1

    def test_1d_array_accepted(self):
        X = _rng_data(c=1).squeeze()
        assert X.ndim == 1
        fc = _quick_fc().fit(X)
        assert fc._enc_in == 1

    def test_different_models_fit(self):
        X = _rng_data()
        for name in ["DLinear", "NLinear", "RNNForecaster"]:
            _quick_fc(name).fit(X)


# ── Forecaster predict ────────────────────────────────────────────────────────


class TestForecasterPredict:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())

    def test_single_window_shape(self):
        x = _rng_data()[-SEQ:]
        y = self.fc.predict(x)
        assert y.shape == (PRED, C)

    def test_longer_context_trimmed(self):
        x = _rng_data()[-100:]  # longer than seq_len
        y = self.fc.predict(x)
        assert y.shape == (PRED, C)

    def test_batch_predict_shape(self):
        X = np.stack([_rng_data()[-SEQ:] for _ in range(4)])  # (4, SEQ, C)
        y = self.fc.predict(X)
        assert y.shape == (4, PRED, C)

    def test_output_is_numpy(self):
        x = _rng_data()[-SEQ:]
        y = self.fc.predict(x)
        assert isinstance(y, np.ndarray)

    def test_output_finite(self):
        x = _rng_data()[-SEQ:]
        y = self.fc.predict(x)
        assert np.isfinite(y).all()

    def test_predict_returns_original_scale(self):
        # With normalize=True, output should be in raw scale:
        # if input mean is ~0, output should not blow up
        X = _rng_data() + 100.0  # shift by 100
        fc = _quick_fc(normalize=True).fit(X)
        y = fc.predict(X[-SEQ:])
        # Rough sanity: predictions should be close-ish to input range
        assert abs(y.mean()) < 200  # raw scale, shifted by ~100

    def test_no_normalize_predict_shape(self):
        fc = _quick_fc(normalize=False).fit(_rng_data())
        y = fc.predict(_rng_data()[-SEQ:])
        assert y.shape == (PRED, C)

    def test_1d_context_accepted(self):
        fc = _quick_fc().fit(_rng_data(c=1))
        y = fc.predict(_rng_data(c=1)[-SEQ:].squeeze())  # 1-D input
        assert y.shape == (PRED, 1)


# ── Forecaster score ──────────────────────────────────────────────────────────


class TestForecasterScore:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())

    def test_returns_dict_with_expected_keys(self):
        result = self.fc.score(_rng_data(n=200))
        assert set(result.keys()) == {"mse", "mae", "rmse"}

    def test_all_values_positive(self):
        result = self.fc.score(_rng_data(n=200))
        for v in result.values():
            assert v >= 0.0

    def test_rmse_equals_sqrt_mse(self):
        result = self.fc.score(_rng_data(n=200))
        assert abs(result["rmse"] - result["mse"] ** 0.5) < 1e-5

    def test_score_too_short_raises(self):
        with pytest.raises(ValueError):
            self.fc.score(np.zeros((5, C)))

    def test_mse_finite(self):
        result = self.fc.score(_rng_data(n=200))
        assert np.isfinite(result["mse"])


# ── Forecaster with nn.Module ─────────────────────────────────────────────────


class TestForecasterWithModule:
    def test_accepts_nn_module(self):
        from torch_timeseries.model import DLinear
        m = DLinear(seq_len=SEQ, pred_len=PRED, enc_in=C)
        fc = Forecaster(m, seq_len=SEQ, pred_len=PRED, epochs=2, verbose=False)
        fc.fit(_rng_data())
        y = fc.predict(_rng_data()[-SEQ:])
        assert y.shape == (PRED, C)


# ── Top-level import ──────────────────────────────────────────────────────────


class TestTopLevelImport:
    def test_forecaster_importable_from_package(self):
        from torch_timeseries import Forecaster as FC
        assert FC is Forecaster

    def test_compare_importable_from_package(self):
        from torch_timeseries import compare as cmp
        assert cmp is compare

    def test_forecaster_repr_correct(self):
        fc = Forecaster("DLinear", seq_len=12, pred_len=4)
        assert "DLinear" in repr(fc)
        assert "seq_len=12" in repr(fc)
        assert "pred_len=4" in repr(fc)


# ── compare() ─────────────────────────────────────────────────────────────────


class TestCompare:
    def test_returns_dict(self):
        X = _rng_data(n=300)
        result = compare(
            ["DLinear", "NLinear"],
            X_train=X[:200], X_test=X[200:],
            seq_len=SEQ, pred_len=PRED, epochs=2, verbose=False,
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {"DLinear", "NLinear"}

    def test_each_entry_has_metrics(self):
        X = _rng_data(n=300)
        result = compare(
            ["DLinear"],
            X_train=X[:200], X_test=X[200:],
            seq_len=SEQ, pred_len=PRED, epochs=2, verbose=False,
        )
        assert "mse" in result["DLinear"]
        assert "mae" in result["DLinear"]
        assert "rmse" in result["DLinear"]

    def test_sorted_by_mse(self):
        X = _rng_data(n=400)
        result = compare(
            ["DLinear", "NLinear", "RNNForecaster"],
            X_train=X[:300], X_test=X[300:],
            seq_len=SEQ, pred_len=PRED, epochs=2, verbose=False,
        )
        mses = [v["mse"] for v in result.values()]
        assert mses == sorted(mses)

    def test_forecaster_instance_accepted(self):
        X = _rng_data(n=300)
        fc = _quick_fc("DLinear")
        result = compare(
            [fc],
            X_train=X[:200], X_test=X[200:],
            seq_len=SEQ, pred_len=PRED, epochs=2, verbose=False,
        )
        assert len(result) == 1
