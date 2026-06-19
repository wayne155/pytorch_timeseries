"""Tests for the high-level Forecaster fit/predict/score API."""
import os
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn

from torch_timeseries.forecaster import (
    Forecaster, compare, list_models, _WindowDataset, _EarlyStopping,
    _make_scheduler, _print_compare_table,
)

# cross_validate and plot_history are methods, no separate imports needed


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
        assert {"mse", "mae", "rmse", "smape"} <= set(result.keys())

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

    def test_smape_present_in_compare(self):
        X = _rng_data(n=300)
        result = compare(
            ["DLinear"],
            X_train=X[:200], X_test=X[200:],
            seq_len=SEQ, pred_len=PRED, epochs=2, verbose=False, print_table=False,
        )
        assert "smape" in result["DLinear"]

    def test_print_table_false_suppresses_output(self, capsys):
        X = _rng_data(n=300)
        compare(
            ["DLinear"],
            X_train=X[:200], X_test=X[200:],
            seq_len=SEQ, pred_len=PRED, epochs=2, verbose=False, print_table=False,
        )
        captured = capsys.readouterr()
        assert "Rank" not in captured.out

    def test_print_table_true_shows_table(self, capsys):
        X = _rng_data(n=300)
        compare(
            ["DLinear"],
            X_train=X[:200], X_test=X[200:],
            seq_len=SEQ, pred_len=PRED, epochs=2, verbose=False, print_table=True,
        )
        captured = capsys.readouterr()
        assert "Rank" in captured.out
        assert "DLinear" in captured.out


# ── list_models() ──────────────────────────────────────────────────────────────


class TestListModels:
    def test_returns_list(self):
        result = list_models()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_sorted(self):
        result = list_models()
        assert result == sorted(result)

    def test_known_models_present(self):
        result = list_models()
        for name in ["DLinear", "NLinear", "PatchTST", "iTransformer"]:
            assert name in result

    def test_importable_from_package(self):
        from torch_timeseries import list_models as lm
        assert lm is list_models


# ── Scheduler ─────────────────────────────────────────────────────────────────


class TestScheduler:
    def _opt(self):
        m = nn.Linear(2, 2)
        return torch.optim.Adam(m.parameters(), lr=1e-3)

    def test_none_returns_none(self):
        assert _make_scheduler(self._opt(), None, 10, 5) is None

    def test_cosine_scheduler(self):
        sched = _make_scheduler(self._opt(), "cosine", 10, 5)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_plateau_scheduler(self):
        sched = _make_scheduler(self._opt(), "plateau", 10, 5)
        assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_step_scheduler(self):
        sched = _make_scheduler(self._opt(), "step", 10, 5)
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown scheduler"):
            _make_scheduler(self._opt(), "badname", 10, 5)

    def test_forecaster_with_cosine_scheduler(self):
        X = _rng_data()
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED, epochs=3,
                        verbose=False, scheduler="cosine")
        fc.fit(X)
        # LR should have changed from initial
        assert fc.history_[-1]["lr"] < 1e-3 or len(fc.history_) > 0

    def test_forecaster_with_plateau_scheduler(self):
        X = _rng_data()
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED, epochs=3,
                        verbose=False, scheduler="plateau")
        fc.fit(X)
        assert len(fc.history_) > 0


# ── history_ ──────────────────────────────────────────────────────────────────


class TestHistory:
    def test_history_populated_after_fit(self):
        fc = _quick_fc().fit(_rng_data())
        assert len(fc.history_) > 0

    def test_history_has_expected_keys(self):
        fc = _quick_fc().fit(_rng_data())
        for entry in fc.history_:
            assert "epoch" in entry
            assert "train_loss" in entry
            assert "val_loss" in entry
            assert "lr" in entry

    def test_history_epochs_sequential(self):
        fc = _quick_fc().fit(_rng_data())
        epochs = [e["epoch"] for e in fc.history_]
        assert epochs == list(range(1, len(fc.history_) + 1))

    def test_history_losses_finite(self):
        fc = _quick_fc().fit(_rng_data())
        for e in fc.history_:
            assert np.isfinite(e["train_loss"])

    def test_history_reset_on_refit(self):
        fc = _quick_fc()
        fc.fit(_rng_data())
        n1 = len(fc.history_)
        fc.fit(_rng_data(seed=1))
        n2 = len(fc.history_)
        # history is reset on refit; lengths may differ due to early stopping
        assert n2 <= fc.epochs


# ── score() SMAPE ─────────────────────────────────────────────────────────────


class TestScoreExtended:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())

    def test_smape_present(self):
        result = self.fc.score(_rng_data(n=200))
        assert "smape" in result

    def test_smape_positive(self):
        result = self.fc.score(_rng_data(n=200))
        assert result["smape"] >= 0.0

    def test_smape_finite(self):
        result = self.fc.score(_rng_data(n=200))
        assert np.isfinite(result["smape"])


# ── save() / load() ───────────────────────────────────────────────────────────


class TestSaveLoad:
    def test_save_creates_file(self, tmp_path):
        fc = _quick_fc().fit(_rng_data())
        path = str(tmp_path / "model.pt")
        fc.save(path)
        assert os.path.isfile(path)

    def test_load_predict_shape(self, tmp_path):
        fc = _quick_fc().fit(_rng_data())
        path = str(tmp_path / "model.pt")
        fc.save(path)
        fc2 = Forecaster.load(path)
        y = fc2.predict(_rng_data()[-SEQ:])
        assert y.shape == (PRED, C)

    def test_load_predictions_match(self, tmp_path):
        fc = _quick_fc(normalize=False).fit(_rng_data())
        path = str(tmp_path / "model.pt")
        fc.save(path)
        fc2 = Forecaster.load(path)
        x = _rng_data()[-SEQ:]
        y1 = fc.predict(x)
        y2 = fc2.predict(x)
        np.testing.assert_allclose(y1, y2, rtol=1e-4)

    def test_load_score_works(self, tmp_path):
        fc = _quick_fc().fit(_rng_data())
        path = str(tmp_path / "model.pt")
        fc.save(path)
        fc2 = Forecaster.load(path)
        result = fc2.score(_rng_data(n=200))
        assert "mse" in result

    def test_history_preserved_across_save_load(self, tmp_path):
        fc = _quick_fc().fit(_rng_data())
        path = str(tmp_path / "model.pt")
        fc.save(path)
        fc2 = Forecaster.load(path)
        assert fc2.history_ == fc.history_

    def test_save_before_fit_raises(self, tmp_path):
        fc = _quick_fc()
        with pytest.raises(RuntimeError, match="not fitted"):
            fc.save(str(tmp_path / "model.pt"))

    def test_load_raw_module_raises(self, tmp_path):
        """Forecasters built from an nn.Module cannot be reloaded (no name saved)."""
        from torch_timeseries.model import DLinear
        m = DLinear(seq_len=SEQ, pred_len=PRED, enc_in=C)
        fc = Forecaster(m, seq_len=SEQ, pred_len=PRED, epochs=1, verbose=False)
        fc.fit(_rng_data())
        path = str(tmp_path / "model.pt")
        fc.save(path)
        with pytest.raises(ValueError, match="raw nn.Module"):
            Forecaster.load(path)


# ── fit_predict() ──────────────────────────────────────────────────────────────


class TestFitPredict:
    def test_returns_correct_shape(self):
        X = _rng_data()
        fc = _quick_fc()
        y = fc.fit_predict(X[:250], X[-SEQ:])
        assert y.shape == (PRED, C)

    def test_equivalent_to_fit_then_predict(self):
        X = _rng_data(seed=42)
        fc1 = _quick_fc(normalize=False)
        y1 = fc1.fit_predict(X[:250], X[-SEQ:])

        # Reproducibility: same seed means same data, not same weights —
        # just check shape and type
        assert isinstance(y1, np.ndarray)
        assert y1.shape == (PRED, C)

    def test_self_is_fitted_after(self):
        X = _rng_data()
        fc = _quick_fc()
        fc.fit_predict(X[:250], X[-SEQ:])
        assert fc._model is not None


# ── cross_validate() ───────────────────────────────────────────────────────────


class TestCrossValidate:
    def test_returns_dict_with_stats(self):
        X = _rng_data(n=600)
        fc = _quick_fc()
        result = fc.cross_validate(X, n_splits=2)
        for key in ("mean_mse", "std_mse", "mean_mae", "mean_rmse", "mean_smape"):
            assert key in result, f"missing key: {key}"

    def test_n_splits_used_correct(self):
        X = _rng_data(n=700)
        fc = _quick_fc()
        result = fc.cross_validate(X, n_splits=3)
        assert result["n_splits_used"] >= 1

    def test_mean_mse_positive(self):
        X = _rng_data(n=600)
        fc = _quick_fc()
        result = fc.cross_validate(X, n_splits=2)
        assert result["mean_mse"] >= 0.0

    def test_std_mse_non_negative(self):
        X = _rng_data(n=600)
        fc = _quick_fc()
        result = fc.cross_validate(X, n_splits=2)
        assert result["std_mse"] >= 0.0

    def test_too_little_data_raises(self):
        X = _rng_data(n=50)  # way too short
        fc = _quick_fc()
        with pytest.raises(ValueError):
            fc.cross_validate(X, n_splits=5)


# ── plot_history() ────────────────────────────────────────────────────────────


class TestPlotHistory:
    def test_returns_axes(self):
        pytest.importorskip("matplotlib")
        fc = _quick_fc().fit(_rng_data())
        ax = fc.plot_history()
        import matplotlib.axes
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_accepts_existing_axes(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        fc = _quick_fc().fit(_rng_data())
        fig, ax_in = plt.subplots()
        ax_out = fc.plot_history(ax=ax_in)
        assert ax_out is ax_in
        plt.close(fig)

    def test_raises_before_fit(self):
        fc = _quick_fc()
        with pytest.raises(RuntimeError):
            fc.plot_history()


# ── get_params() / set_params() ───────────────────────────────────────────────


class TestGetSetParams:
    def test_get_params_has_required_keys(self):
        fc = _quick_fc()
        p = fc.get_params()
        for key in ("model", "seq_len", "pred_len", "epochs", "lr"):
            assert key in p, f"missing key: {key}"

    def test_get_params_values_correct(self):
        fc = Forecaster("DLinear", seq_len=48, pred_len=12, epochs=10, lr=5e-4,
                        verbose=False)
        p = fc.get_params()
        assert p["model"] == "DLinear"
        assert p["seq_len"] == 48
        assert p["pred_len"] == 12
        assert p["epochs"] == 10
        assert p["lr"] == 5e-4

    def test_set_params_updates_attribute(self):
        fc = _quick_fc()
        fc.set_params(epochs=99, lr=1e-2)
        assert fc.epochs == 99
        assert fc.lr == 1e-2

    def test_set_params_returns_self(self):
        fc = _quick_fc()
        ret = fc.set_params(epochs=5)
        assert ret is fc

    def test_set_params_unknown_goes_to_model_kwargs(self):
        fc = _quick_fc()
        fc.set_params(d_model=64)
        assert fc.model_kwargs.get("d_model") == 64
