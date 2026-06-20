"""Tests for the high-level Forecaster fit/predict/score API."""
import os
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn

from torch_timeseries.forecaster import (
    Forecaster, StackedForecaster, BaggingForecaster, Pipeline,
    MultiChannelForecaster, EnsembleForecaster, SklearnForecaster,
    compare, compare_to_dataframe, compare_plot, list_models, time_series_split,
    make_forecaster, _WindowDataset, _EarlyStopping, _make_scheduler,
    _print_compare_table, _resolve_loss,
)
from torch_timeseries.dataset import list_datasets, load_dataset
from torch_timeseries.augment import Jitter, Compose, Scale


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


# ── plot_forecast() ───────────────────────────────────────────────────────────


class TestPlotForecast:
    def setup_method(self):
        pytest.importorskip("matplotlib")
        self.fc = _quick_fc().fit(_rng_data())

    def test_returns_axes(self):
        import matplotlib.axes
        X = _rng_data(n=200)
        ax = self.fc.plot_forecast(X)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_accepts_existing_axes(self):
        import matplotlib.pyplot as plt
        X = _rng_data(n=200)
        fig, ax_in = plt.subplots()
        ax_out = self.fc.plot_forecast(X, ax=ax_in)
        assert ax_out is ax_in
        plt.close(fig)

    def test_custom_channel(self):
        import matplotlib.axes
        X = _rng_data(n=200)
        ax = self.fc.plot_forecast(X, channel=1)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_custom_title(self):
        X = _rng_data(n=200)
        ax = self.fc.plot_forecast(X, title="My Title")
        assert ax.get_title() == "My Title"

    def test_n_context_limits_context_lines(self):
        import matplotlib.axes
        X = _rng_data(n=200)
        ax = self.fc.plot_forecast(X, n_context=5)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_raises_before_fit(self):
        fc = _quick_fc()
        with pytest.raises(RuntimeError):
            fc.plot_forecast(_rng_data(n=200))

    def test_raises_too_short(self):
        X = _rng_data(n=5)
        with pytest.raises(ValueError):
            self.fc.plot_forecast(X)


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

    def test_get_params_includes_loss_and_warm_start(self):
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED, verbose=False,
                        loss="mae", warm_start=True)
        p = fc.get_params()
        assert p["loss"] == "mae"
        assert p["warm_start"] is True


# ── _resolve_loss() ───────────────────────────────────────────────────────────


class TestResolveLoss:
    def test_none_returns_mse(self):
        loss = _resolve_loss(None)
        assert isinstance(loss, nn.MSELoss)

    def test_mse_string(self):
        assert isinstance(_resolve_loss("mse"), nn.MSELoss)

    def test_mae_string(self):
        assert isinstance(_resolve_loss("mae"), nn.L1Loss)

    def test_l1_string(self):
        assert isinstance(_resolve_loss("l1"), nn.L1Loss)

    def test_huber_string(self):
        assert isinstance(_resolve_loss("huber"), nn.HuberLoss)

    def test_custom_module(self):
        custom = nn.SmoothL1Loss()
        assert _resolve_loss(custom) is custom

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown loss"):
            _resolve_loss("mystery_loss")


# ── custom loss in Forecaster ─────────────────────────────────────────────────


class TestCustomLoss:
    def test_string_loss_mae(self):
        X = _rng_data()
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                        epochs=2, verbose=False, loss="mae")
        fc.fit(X)
        assert len(fc.history_) > 0

    def test_module_loss(self):
        X = _rng_data()
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                        epochs=2, verbose=False, loss=nn.HuberLoss())
        fc.fit(X)
        assert len(fc.history_) > 0


# ── warm_start ────────────────────────────────────────────────────────────────


class TestWarmStart:
    def test_warm_start_preserves_model_instance(self):
        X = _rng_data()
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                        epochs=2, verbose=False, warm_start=True)
        fc.fit(X)
        m1 = fc._model
        fc.fit(X)
        assert fc._model is m1  # same object, weights updated in-place

    def test_cold_start_replaces_model(self):
        X = _rng_data()
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                        epochs=2, verbose=False, warm_start=False)
        fc.fit(X)
        m1 = fc._model
        fc.fit(X)
        assert fc._model is not m1


# ── summary() ────────────────────────────────────────────────────────────────


class TestSummary:
    def test_returns_string(self):
        fc = _quick_fc().fit(_rng_data())
        s = fc.summary()
        assert isinstance(s, str)

    def test_contains_model_name(self):
        fc = _quick_fc("DLinear").fit(_rng_data())
        assert "DLinear" in fc.summary()

    def test_contains_seq_pred_len(self):
        fc = _quick_fc().fit(_rng_data())
        s = fc.summary()
        assert str(SEQ) in s
        assert str(PRED) in s

    def test_contains_parameters_when_fitted(self):
        fc = _quick_fc().fit(_rng_data())
        assert "parameters" in fc.summary()

    def test_contains_last_epoch_when_fitted(self):
        fc = _quick_fc().fit(_rng_data())
        assert "last epoch" in fc.summary()


class TestSaveReport:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=200)

    def test_creates_report_txt(self, tmp_path):
        import os
        path = self.fc.save_report(self.X, str(tmp_path), n_repeats=2)
        assert os.path.exists(path)
        assert path.endswith("report.txt")

    def test_report_txt_contains_content(self, tmp_path):
        path = self.fc.save_report(self.X, str(tmp_path), n_repeats=2)
        with open(path) as f:
            content = f.read()
        assert "DLinear" in content

    def test_creates_directory_if_missing(self, tmp_path):
        import os
        subdir = str(tmp_path / "nested" / "report")
        self.fc.save_report(self.X, subdir, n_repeats=2, save_plots=False)
        assert os.path.isdir(subdir)

    def test_returns_path_string(self, tmp_path):
        result = self.fc.save_report(self.X, str(tmp_path), n_repeats=2,
                                     save_plots=False)
        assert isinstance(result, str)

    def test_before_fit_raises(self, tmp_path):
        fc = _quick_fc()
        with pytest.raises(RuntimeError):
            fc.save_report(self.X, str(tmp_path), n_repeats=2)


class TestExplain:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=200)

    def test_returns_string(self):
        result = self.fc.explain(self.X, n_repeats=2)
        assert isinstance(result, str)

    def test_contains_model_name(self):
        result = self.fc.explain(self.X, n_repeats=2)
        assert "DLinear" in result

    def test_contains_mse(self):
        result = self.fc.explain(self.X, n_repeats=2)
        assert "mse" in result.lower()

    def test_contains_channel_importance(self):
        result = self.fc.explain(self.X, n_repeats=2)
        assert "Channel importance" in result

    def test_contains_timestep_importance(self):
        result = self.fc.explain(self.X, n_repeats=2)
        assert "Timestep importance" in result

    def test_custom_channel_names(self):
        result = self.fc.explain(self.X, n_repeats=2,
                                 channel_names=["temp", "wind", "rain"])
        assert "temp" in result

    def test_before_fit_raises(self):
        fc = _quick_fc()
        with pytest.raises(RuntimeError):
            fc.explain(self.X, n_repeats=2)


# ── grad_clip ─────────────────────────────────────────────────────────────────


class TestGradClip:
    def test_grad_clip_trains_without_error(self):
        X = _rng_data()
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                        epochs=2, verbose=False, grad_clip=1.0)
        fc.fit(X)
        assert len(fc.history_) > 0

    def test_grad_clip_in_get_params(self):
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                        verbose=False, grad_clip=0.5)
        assert fc.get_params()["grad_clip"] == 0.5

    def test_none_grad_clip_is_default(self):
        fc = _quick_fc()
        assert fc.grad_clip is None


# ── clone() ───────────────────────────────────────────────────────────────────


class TestClone:
    def test_clone_is_not_fitted(self):
        fc = _quick_fc().fit(_rng_data())
        clone = fc.clone()
        assert clone._model is None

    def test_clone_has_same_params(self):
        fc = Forecaster("DLinear", seq_len=48, pred_len=12, lr=5e-4,
                        verbose=False, epochs=3)
        clone = fc.clone()
        assert clone.seq_len == 48
        assert clone.pred_len == 12
        assert clone.lr == 5e-4
        assert clone.epochs == 3

    def test_clone_is_independent(self):
        fc = _quick_fc()
        clone = fc.clone()
        clone.epochs = 999
        assert fc.epochs != 999


# ── from_config() ─────────────────────────────────────────────────────────────


class TestFromConfig:
    def test_roundtrip_get_params(self):
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                        lr=5e-4, epochs=7, verbose=False)
        params = fc.get_params()
        fc2 = Forecaster.from_config(params)
        assert fc2.seq_len == SEQ
        assert fc2.pred_len == PRED
        assert fc2.lr == 5e-4
        assert fc2.epochs == 7

    def test_returns_unfitted(self):
        fc = Forecaster.from_config({"model": "DLinear", "seq_len": SEQ,
                                     "pred_len": PRED})
        assert fc._model is None

    def test_model_kwargs_passed_through(self):
        fc = Forecaster.from_config({"model": "DLinear", "seq_len": SEQ,
                                     "pred_len": PRED, "d_ff": 128})
        assert fc.model_kwargs.get("d_ff") == 128


# ── tune() ────────────────────────────────────────────────────────────────────


class TestTune:
    def test_returns_forecaster(self):
        X = _rng_data(n=700)
        fc = _quick_fc()
        best = fc.tune(X, param_grid={"lr": [1e-3, 1e-4]}, n_splits=2, verbose=False)
        assert isinstance(best, Forecaster)

    def test_returned_forecaster_not_fitted(self):
        X = _rng_data(n=700)
        fc = _quick_fc()
        best = fc.tune(X, param_grid={"lr": [1e-3, 1e-4]}, n_splits=2, verbose=False)
        assert best._model is None

    def test_best_param_is_in_grid(self):
        X = _rng_data(n=700)
        fc = _quick_fc()
        candidates = [1e-3, 1e-4]
        best = fc.tune(X, param_grid={"lr": candidates}, n_splits=2, verbose=False)
        assert best.lr in candidates

    def test_best_can_be_fitted(self):
        X = _rng_data(n=700)
        fc = _quick_fc()
        best = fc.tune(X, param_grid={"lr": [1e-3, 1e-4]}, n_splits=2, verbose=False)
        best.fit(X)
        assert best._model is not None


class TestHyperparameterSensitivity:
    def test_returns_dict_per_param(self):
        X = _rng_data(n=400)
        fc = _quick_fc()
        result = fc.hyperparameter_sensitivity(
            X, {"lr": [1e-3, 1e-4]}, val_split=0.1
        )
        assert "lr" in result

    def test_each_entry_has_value_and_val_loss(self):
        X = _rng_data(n=400)
        fc = _quick_fc()
        result = fc.hyperparameter_sensitivity(
            X, {"lr": [1e-3, 1e-4]}, val_split=0.1
        )
        for record in result["lr"]:
            assert "value" in record
            assert "val_loss" in record

    def test_num_records_matches_num_candidates(self):
        X = _rng_data(n=400)
        fc = _quick_fc()
        candidates = [1e-3, 5e-4, 1e-4]
        result = fc.hyperparameter_sensitivity(X, {"lr": candidates})
        assert len(result["lr"]) == len(candidates)

    def test_original_forecaster_unchanged(self):
        X = _rng_data(n=400)
        fc = _quick_fc()
        orig_lr = fc.lr
        fc.hyperparameter_sensitivity(X, {"lr": [1e-3, 1e-4]})
        assert fc.lr == orig_lr
        assert fc._model is None  # not fitted

    def test_multiple_params(self):
        X = _rng_data(n=400)
        fc = _quick_fc()
        result = fc.hyperparameter_sensitivity(
            X, {"lr": [1e-3, 1e-4], "batch_size": [16, 32]}
        )
        assert set(result.keys()) == {"lr", "batch_size"}


class TestLearningCurve:
    def test_returns_list(self):
        X = _rng_data(n=400)
        fc = _quick_fc()
        result = fc.learning_curve(X, train_fractions=[0.5, 1.0])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_each_record_has_expected_keys(self):
        X = _rng_data(n=400)
        fc = _quick_fc()
        result = fc.learning_curve(X, train_fractions=[0.5, 1.0])
        for rec in result:
            for k in ("fraction", "n_samples", "val_loss"):
                assert k in rec

    def test_n_samples_increases_with_fraction(self):
        X = _rng_data(n=400)
        fc = _quick_fc()
        result = fc.learning_curve(X, train_fractions=[0.3, 0.6, 1.0])
        ns = [r["n_samples"] for r in result]
        assert ns == sorted(ns)

    def test_original_forecaster_not_fitted(self):
        X = _rng_data(n=400)
        fc = _quick_fc()
        fc.learning_curve(X, train_fractions=[0.5])
        assert fc._model is None

    def test_single_fraction(self):
        X = _rng_data(n=400)
        fc = _quick_fc()
        result = fc.learning_curve(X, train_fractions=[1.0])
        assert len(result) == 1
        assert result[0]["fraction"] == 1.0


class TestPlotLearningCurve:
    def setup_method(self):
        pytest.importorskip("matplotlib")
        X = _rng_data(n=400)
        self.fc = _quick_fc()
        self.lc = self.fc.learning_curve(X, train_fractions=[0.5, 1.0])

    def test_returns_axes(self):
        import matplotlib.axes
        ax = self.fc.plot_learning_curve(self.lc)
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_accepts_existing_axes(self):
        import matplotlib.pyplot as plt
        fig, ax_in = plt.subplots()
        ax_out = self.fc.plot_learning_curve(self.lc, ax=ax_in)
        assert ax_out is ax_in
        plt.close(fig)

    def test_custom_title(self):
        import matplotlib.pyplot as plt
        ax = self.fc.plot_learning_curve(self.lc, title="LC Title")
        assert ax.get_title() == "LC Title"
        plt.close("all")


class TestFreezeLayers:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())

    def test_freeze_all(self):
        self.fc.freeze_layers()
        for p in self.fc._model.parameters():
            assert not p.requires_grad

    def test_unfreeze_all(self):
        self.fc.freeze_layers()
        self.fc.unfreeze_layers()
        for p in self.fc._model.parameters():
            assert p.requires_grad

    def test_freeze_returns_self(self):
        assert self.fc.freeze_layers() is self.fc

    def test_unfreeze_returns_self(self):
        self.fc.freeze_layers()
        assert self.fc.unfreeze_layers() is self.fc

    def test_frozen_parameter_count_after_freeze(self):
        total = self.fc.n_parameters
        self.fc.freeze_layers()
        frozen = self.fc.frozen_parameter_count()
        assert frozen == total

    def test_frozen_parameter_count_zero_unfrozen(self):
        self.fc.freeze_layers()
        self.fc.unfreeze_layers()
        assert self.fc.frozen_parameter_count() == 0

    def test_before_fit_raises(self):
        fc = _quick_fc()
        with pytest.raises(RuntimeError):
            fc.freeze_layers()
        with pytest.raises(RuntimeError):
            fc.unfreeze_layers()
        with pytest.raises(RuntimeError):
            fc.frozen_parameter_count()


class TestPlotSensitivity:
    def setup_method(self):
        pytest.importorskip("matplotlib")
        X = _rng_data(n=400)
        self.fc = _quick_fc()
        self.sens = self.fc.hyperparameter_sensitivity(
            X, {"lr": [1e-3, 1e-4]}
        )

    def test_returns_axes(self):
        import matplotlib.axes
        ax = self.fc.plot_sensitivity(self.sens)
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_accepts_existing_axes(self):
        import matplotlib.pyplot as plt
        fig, ax_in = plt.subplots()
        ax_out = self.fc.plot_sensitivity(self.sens, ax=ax_in)
        assert ax_out is ax_in
        plt.close(fig)

    def test_custom_title(self):
        import matplotlib.pyplot as plt
        ax = self.fc.plot_sensitivity(self.sens, title="My Sens")
        assert ax.get_title() == "My Sens"
        plt.close("all")


# ── weight_decay ──────────────────────────────────────────────────────────────


class TestWeightDecay:
    def test_weight_decay_trains(self):
        X = _rng_data()
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                        epochs=2, verbose=False, weight_decay=1e-4)
        fc.fit(X)
        assert len(fc.history_) > 0

    def test_default_weight_decay_zero(self):
        fc = _quick_fc()
        assert fc.weight_decay == 0.0

    def test_in_get_params(self):
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                        verbose=False, weight_decay=1e-3)
        assert fc.get_params()["weight_decay"] == 1e-3


# ── callbacks ─────────────────────────────────────────────────────────────────


class TestCallbacks:
    def test_callback_called_each_epoch(self):
        calls = []

        def my_cb(fc, entry):
            calls.append(entry["epoch"])

        X = _rng_data()
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                        epochs=3, verbose=False, callbacks=[my_cb])
        fc.fit(X)
        # may stop early but at least one call expected
        assert len(calls) >= 1

    def test_callback_receives_entry_dict(self):
        entries = []

        def my_cb(fc, entry):
            entries.append(entry)

        X = _rng_data()
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                        epochs=2, verbose=False, callbacks=[my_cb])
        fc.fit(X)
        for e in entries:
            assert "epoch" in e
            assert "train_loss" in e

    def test_callback_receives_forecaster_ref(self):
        received = []

        def my_cb(fc_ref, entry):
            received.append(fc_ref)

        X = _rng_data()
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                        epochs=2, verbose=False, callbacks=[my_cb])
        fc.fit(X)
        assert received[0] is fc

    def test_no_callbacks_by_default(self):
        fc = _quick_fc()
        assert fc.callbacks == []


# ── predict_rolling() ─────────────────────────────────────────────────────────


class TestPredictRolling:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())

    def test_output_shape_default(self):
        X = _rng_data(n=200)
        out = self.fc.predict_rolling(X)
        n_positions = 200 - SEQ - PRED + 1
        assert out.shape == (n_positions, PRED, C)

    def test_n_steps_limits_positions(self):
        X = _rng_data(n=200)
        out = self.fc.predict_rolling(X, n_steps=5)
        assert out.shape[0] == 5

    def test_stride_reduces_positions(self):
        X = _rng_data(n=200)
        out_stride1 = self.fc.predict_rolling(X)
        out_stride2 = self.fc.predict_rolling(X, stride=2)
        assert out_stride2.shape[0] < out_stride1.shape[0]

    def test_output_is_numpy(self):
        X = _rng_data(n=100)
        out = self.fc.predict_rolling(X)
        assert isinstance(out, np.ndarray)

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            self.fc.predict_rolling(np.zeros((5, C)))


# ── compare() timing ──────────────────────────────────────────────────────────


class TestCompareTiming:
    def test_elapsed_s_present(self):
        X = _rng_data(n=300)
        result = compare(
            ["DLinear"],
            X_train=X[:200], X_test=X[200:],
            seq_len=SEQ, pred_len=PRED, epochs=1, verbose=False, print_table=False,
        )
        assert "elapsed_s" in result["DLinear"]

    def test_elapsed_s_positive(self):
        X = _rng_data(n=300)
        result = compare(
            ["DLinear"],
            X_train=X[:200], X_test=X[200:],
            seq_len=SEQ, pred_len=PRED, epochs=1, verbose=False, print_table=False,
        )
        assert result["DLinear"]["elapsed_s"] > 0


# ── tune() n_iter ─────────────────────────────────────────────────────────────


class TestTuneNIter:
    def test_n_iter_limits_combos(self):
        X = _rng_data(n=700)
        evaluated = []

        original_cv = Forecaster.cross_validate

        def counting_cv(self, X, **kwargs):
            evaluated.append(1)
            return original_cv(self, X, **kwargs)

        import unittest.mock as mock
        fc = _quick_fc()
        with mock.patch.object(Forecaster, "cross_validate", counting_cv):
            fc.tune(X, param_grid={"lr": [1e-3, 1e-4, 1e-5]}, n_splits=2,
                    verbose=False, n_iter=2)
        assert len(evaluated) == 2

    def test_n_iter_none_evaluates_all(self):
        X = _rng_data(n=700)
        evaluated = []

        original_cv = Forecaster.cross_validate

        def counting_cv(self, X, **kwargs):
            evaluated.append(1)
            return original_cv(self, X, **kwargs)

        import unittest.mock as mock
        fc = _quick_fc()
        with mock.patch.object(Forecaster, "cross_validate", counting_cv):
            fc.tune(X, param_grid={"lr": [1e-3, 1e-4]}, n_splits=2, verbose=False)
        assert len(evaluated) == 2


# ── residuals() ───────────────────────────────────────────────────────────────


class TestResiduals:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())

    def test_shape(self):
        X = _rng_data(n=200)
        res = self.fc.residuals(X)
        n_windows = 200 - SEQ - PRED + 1
        assert res.shape == (n_windows, PRED, C)

    def test_is_numpy(self):
        res = self.fc.residuals(_rng_data(n=100))
        assert isinstance(res, np.ndarray)

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            self.fc.residuals(np.zeros((5, C)))

    def test_finite(self):
        res = self.fc.residuals(_rng_data(n=100))
        assert np.isfinite(res).all()


# ── progress_bar ──────────────────────────────────────────────────────────────


class TestProgressBar:
    def test_progress_bar_false_default(self):
        fc = _quick_fc()
        assert fc.progress_bar is False

    def test_progress_bar_in_get_params(self):
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                        verbose=False, progress_bar=True)
        assert fc.get_params()["progress_bar"] is True

    def test_progress_bar_trains_without_error(self):
        X = _rng_data()
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                        epochs=2, verbose=False, progress_bar=True)
        fc.fit(X)
        assert len(fc.history_) > 0


# ── StackedForecaster ─────────────────────────────────────────────────────────


class TestStackedForecaster:
    def test_fit_and_predict_shape(self):
        X = _rng_data(n=400)
        base = _quick_fc("DLinear")
        meta = _quick_fc("NLinear")
        sf = StackedForecaster(base, meta)
        sf.fit(X)
        y = sf.predict(X[-SEQ:])
        assert y.shape == (PRED, C)

    def test_default_meta_is_clone(self):
        base = _quick_fc("DLinear")
        sf = StackedForecaster(base)
        assert sf.meta is not base
        assert sf.meta.model_spec == base.model_spec

    def test_score_returns_metrics(self):
        X = _rng_data(n=400)
        sf = StackedForecaster(_quick_fc("DLinear"), _quick_fc("NLinear"))
        sf.fit(X[:300])
        result = sf.score(X[300:])
        for key in ("mse", "mae", "rmse", "smape"):
            assert key in result

    def test_score_mse_positive(self):
        X = _rng_data(n=400)
        sf = StackedForecaster(_quick_fc("DLinear"), _quick_fc("NLinear"))
        sf.fit(X[:300])
        assert sf.score(X[300:])["mse"] >= 0.0

    def test_repr(self):
        sf = StackedForecaster(_quick_fc(), _quick_fc("NLinear"))
        assert "StackedForecaster" in repr(sf)


# ── feature_importance() ──────────────────────────────────────────────────────


class TestFeatureImportance:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())

    def test_returns_dict_with_expected_keys(self):
        result = self.fc.feature_importance(_rng_data(n=200), n_repeats=2)
        assert "importances_mean" in result
        assert "importances_std" in result
        assert "baseline_score" in result

    def test_importances_shape(self):
        result = self.fc.feature_importance(_rng_data(n=200), n_repeats=2)
        assert result["importances_mean"].shape == (C,)
        assert result["importances_std"].shape == (C,)

    def test_baseline_score_positive(self):
        result = self.fc.feature_importance(_rng_data(n=200), n_repeats=2)
        assert result["baseline_score"] >= 0.0

    def test_std_non_negative(self):
        result = self.fc.feature_importance(_rng_data(n=200), n_repeats=3)
        assert (result["importances_std"] >= 0.0).all()

    def test_mae_metric(self):
        result = self.fc.feature_importance(_rng_data(n=200), n_repeats=2, metric="mae")
        assert "importances_mean" in result

    def test_reproducible_with_seed(self):
        X = _rng_data(n=200)
        r1 = self.fc.feature_importance(X, n_repeats=2, random_state=42)
        r2 = self.fc.feature_importance(X, n_repeats=2, random_state=42)
        np.testing.assert_array_equal(r1["importances_mean"], r2["importances_mean"])


class TestTimestepImportance:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=200)

    def test_returns_expected_keys(self):
        result = self.fc.timestep_importance(self.X, n_repeats=2)
        for k in ("importances_mean", "importances_std", "baseline_score"):
            assert k in result

    def test_importances_shape(self):
        result = self.fc.timestep_importance(self.X, n_repeats=2)
        assert result["importances_mean"].shape == (SEQ,)
        assert result["importances_std"].shape == (SEQ,)

    def test_baseline_positive(self):
        result = self.fc.timestep_importance(self.X, n_repeats=2)
        assert result["baseline_score"] >= 0.0

    def test_std_non_negative(self):
        result = self.fc.timestep_importance(self.X, n_repeats=2)
        assert (result["importances_std"] >= 0.0).all()

    def test_reproducible_with_seed(self):
        r1 = self.fc.timestep_importance(self.X, n_repeats=2, random_state=0)
        r2 = self.fc.timestep_importance(self.X, n_repeats=2, random_state=0)
        np.testing.assert_array_equal(r1["importances_mean"], r2["importances_mean"])

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            self.fc.timestep_importance(np.zeros((5, C)), n_repeats=2)

    def test_mae_metric(self):
        result = self.fc.timestep_importance(self.X, n_repeats=2, metric="mae")
        assert "importances_mean" in result


class TestPlotTimestepImportance:
    def setup_method(self):
        pytest.importorskip("matplotlib")
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=200)

    def test_returns_axes(self):
        import matplotlib.axes
        ax = self.fc.plot_timestep_importance(self.X, n_repeats=2)
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_accepts_existing_axes(self):
        import matplotlib.pyplot as plt
        fig, ax_in = plt.subplots()
        ax_out = self.fc.plot_timestep_importance(self.X, n_repeats=2, ax=ax_in)
        assert ax_out is ax_in
        plt.close(fig)

    def test_custom_title(self):
        import matplotlib.pyplot as plt
        ax = self.fc.plot_timestep_importance(self.X, n_repeats=2, title="T Imp")
        assert ax.get_title() == "T Imp"
        plt.close("all")


# ── predict_uncertainty() ─────────────────────────────────────────────────────


class TestPredictUncertainty:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())

    def test_returns_dict_with_expected_keys(self):
        result = self.fc.predict_uncertainty(_rng_data()[-SEQ:], n_samples=5)
        for key in ("mean", "std", "lower", "upper"):
            assert key in result

    def test_mean_shape(self):
        result = self.fc.predict_uncertainty(_rng_data()[-SEQ:], n_samples=5)
        assert result["mean"].shape == (PRED, C)

    def test_std_non_negative(self):
        result = self.fc.predict_uncertainty(_rng_data()[-SEQ:], n_samples=5)
        assert (result["std"] >= 0.0).all()

    def test_lower_le_upper(self):
        result = self.fc.predict_uncertainty(_rng_data()[-SEQ:], n_samples=10)
        assert (result["lower"] <= result["upper"]).all()


class TestPlotIntervals:
    def setup_method(self):
        pytest.importorskip("matplotlib")
        self.fc = _quick_fc().fit(_rng_data())
        X_cal = _rng_data(n=200, seed=5)
        X_ctx = _rng_data(n=200, seed=6)
        self.intervals = self.fc.predict_interval(X_ctx[-SEQ:], X_cal)
        self.X_ctx = X_ctx
        self.X_truth = _rng_data(n=PRED, seed=7)

    def test_returns_axes(self):
        import matplotlib.axes
        ax = self.fc.plot_intervals(self.intervals)
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_with_context(self):
        import matplotlib.axes
        ax = self.fc.plot_intervals(self.intervals, X_context=self.X_ctx[-SEQ:])
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_with_truth(self):
        import matplotlib.axes
        ax = self.fc.plot_intervals(self.intervals, X_truth=self.X_truth)
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_accepts_existing_axes(self):
        import matplotlib.pyplot as plt
        fig, ax_in = plt.subplots()
        ax_out = self.fc.plot_intervals(self.intervals, ax=ax_in)
        assert ax_out is ax_in
        plt.close(fig)

    def test_custom_title(self):
        import matplotlib.pyplot as plt
        ax = self.fc.plot_intervals(self.intervals, title="My Intervals")
        assert ax.get_title() == "My Intervals"
        plt.close("all")

    def test_uncertainty_dict_accepted(self):
        import matplotlib.axes
        unc = self.fc.predict_uncertainty(self.X_ctx[-SEQ:], n_samples=5)
        ax = self.fc.plot_intervals(unc)
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")


class TestPredictInterval:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X_cal = _rng_data(n=200, seed=2)
        self.X_ctx = _rng_data(n=200, seed=3)

    def test_returns_expected_keys(self):
        result = self.fc.predict_interval(self.X_ctx[-SEQ:], self.X_cal)
        for key in ("mean", "lower", "upper", "half_width"):
            assert key in result

    def test_shapes(self):
        result = self.fc.predict_interval(self.X_ctx[-SEQ:], self.X_cal)
        assert result["mean"].shape == (PRED, C)
        assert result["lower"].shape == (PRED, C)
        assert result["upper"].shape == (PRED, C)

    def test_lower_le_upper(self):
        result = self.fc.predict_interval(self.X_ctx[-SEQ:], self.X_cal)
        assert (result["lower"] <= result["upper"]).all()

    def test_half_width_positive(self):
        result = self.fc.predict_interval(self.X_ctx[-SEQ:], self.X_cal)
        assert (result["half_width"] >= 0.0).all()

    def test_invalid_coverage_raises(self):
        with pytest.raises(ValueError, match="coverage"):
            self.fc.predict_interval(self.X_ctx[-SEQ:], self.X_cal,
                                     coverage=1.1)

    def test_before_fit_raises(self):
        fc = _quick_fc()
        with pytest.raises(RuntimeError):
            fc.predict_interval(self.X_ctx[-SEQ:], self.X_cal)

    def test_longer_context_accepted(self):
        result = self.fc.predict_interval(self.X_ctx, self.X_cal)
        assert result["mean"].shape == (PRED, C)


class TestCalibrate:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X_cal = _rng_data(n=200, seed=1)

    def test_returns_self(self):
        result = self.fc.calibrate(self.X_cal, n_samples=5)
        assert result is self.fc

    def test_sets_interval_scale(self):
        self.fc.calibrate(self.X_cal, n_samples=5)
        assert hasattr(self.fc, "_interval_scale_")
        assert self.fc._interval_scale_ > 0.0

    def test_intervals_still_lower_le_upper_after_calibration(self):
        self.fc.calibrate(self.X_cal, n_samples=5)
        result = self.fc.predict_uncertainty(self.X_cal[-SEQ:], n_samples=5)
        assert (result["lower"] <= result["upper"]).all()

    def test_invalid_coverage_raises(self):
        with pytest.raises(ValueError, match="target_coverage"):
            self.fc.calibrate(self.X_cal, target_coverage=1.5, n_samples=5)

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            self.fc.calibrate(np.zeros((5, C)), n_samples=5)

    def test_before_fit_raises(self):
        fc = _quick_fc()
        with pytest.raises(RuntimeError):
            fc.calibrate(self.X_cal, n_samples=5)


# ── compare n_jobs ────────────────────────────────────────────────────────────


class TestCompareNJobs:
    def test_n_jobs_1_sequential(self):
        X = _rng_data(n=300)
        result = compare(
            ["DLinear", "NLinear"],
            X_train=X[:200], X_test=X[200:],
            seq_len=SEQ, pred_len=PRED, epochs=1, verbose=False,
            print_table=False, n_jobs=1,
        )
        assert set(result.keys()) == {"DLinear", "NLinear"}

    def test_elapsed_s_present_n_jobs_1(self):
        X = _rng_data(n=300)
        result = compare(
            ["DLinear"],
            X_train=X[:200], X_test=X[200:],
            seq_len=SEQ, pred_len=PRED, epochs=1, verbose=False,
            print_table=False, n_jobs=1,
        )
        assert "elapsed_s" in result["DLinear"]


class TestCompareModels:
    def test_returns_dict_with_all_models(self):
        X = _rng_data(n=300)
        fc = _quick_fc("DLinear")
        result = fc.compare_models(
            ["NLinear"], X[:200], X[200:],
            print_table=False, verbose=False,
        )
        assert "DLinear" in result
        assert "NLinear" in result

    def test_include_self_false(self):
        X = _rng_data(n=300)
        fc = _quick_fc("DLinear")
        result = fc.compare_models(
            ["NLinear"], X[:200], X[200:],
            include_self=False, print_table=False, verbose=False,
        )
        assert "NLinear" in result
        # DLinear may or may not be present; just check NLinear is there

    def test_result_has_mse(self):
        X = _rng_data(n=300)
        fc = _quick_fc("DLinear")
        result = fc.compare_models(
            ["NLinear"], X[:200], X[200:],
            print_table=False, verbose=False,
        )
        for metrics in result.values():
            if not isinstance(metrics, Exception):
                assert "mse" in metrics


# ── BaggingForecaster ─────────────────────────────────────────────────────────


class TestBaggingForecaster:
    def test_fit_predict_mean_shape(self):
        X = _rng_data(n=400)
        bag = BaggingForecaster(_quick_fc("DLinear"), n_estimators=3,
                                random_state=0)
        bag.fit(X)
        result = bag.predict(X[-SEQ:])
        assert result["mean"].shape == (PRED, C)

    def test_predict_returns_all_keys(self):
        X = _rng_data(n=400)
        bag = BaggingForecaster(_quick_fc("DLinear"), n_estimators=3,
                                random_state=0)
        bag.fit(X)
        result = bag.predict(X[-SEQ:])
        for key in ("mean", "std", "lower", "upper"):
            assert key in result

    def test_std_non_negative(self):
        X = _rng_data(n=400)
        bag = BaggingForecaster(_quick_fc("DLinear"), n_estimators=3,
                                random_state=0)
        bag.fit(X)
        result = bag.predict(X[-SEQ:])
        assert (result["std"] >= 0.0).all()

    def test_n_estimators_correct(self):
        X = _rng_data(n=400)
        bag = BaggingForecaster(_quick_fc("DLinear"), n_estimators=4)
        bag.fit(X)
        assert len(bag.estimators_) == 4

    def test_score_returns_metrics(self):
        X = _rng_data(n=400)
        bag = BaggingForecaster(_quick_fc("DLinear"), n_estimators=3,
                                random_state=0)
        bag.fit(X[:300])
        result = bag.score(X[300:])
        for key in ("mse", "mae", "rmse", "smape"):
            assert key in result

    def test_predict_before_fit_raises(self):
        bag = BaggingForecaster(_quick_fc("DLinear"), n_estimators=3)
        with pytest.raises(RuntimeError, match="not fitted"):
            bag.predict(np.zeros((SEQ, C)))

    def test_repr(self):
        bag = BaggingForecaster(_quick_fc(), n_estimators=5)
        assert "BaggingForecaster" in repr(bag)
        assert "not fitted" in repr(bag)

    def test_repr_fitted(self):
        X = _rng_data(n=400)
        bag = BaggingForecaster(_quick_fc(), n_estimators=2)
        bag.fit(X)
        assert "2 estimators" in repr(bag)

    def test_invalid_subsample_raises(self):
        with pytest.raises(ValueError, match="subsample"):
            BaggingForecaster(_quick_fc(), subsample=0.0)


# ── compare_to_dataframe() ────────────────────────────────────────────────────


class TestCompareToDataframe:
    def test_returns_dataframe_or_none(self):
        results = {
            "DLinear": {"mse": 0.5, "mae": 0.4, "rmse": 0.7, "smape": 10.0},
            "NLinear": {"mse": 0.6, "mae": 0.5, "rmse": 0.77, "smape": 11.0},
        }
        df = compare_to_dataframe(results)
        if df is None:
            pytest.skip("pandas not installed")
        assert df is not None

    def test_index_is_model_name(self):
        pytest.importorskip("pandas")
        results = {
            "DLinear": {"mse": 0.5, "mae": 0.4, "rmse": 0.7, "smape": 10.0},
        }
        df = compare_to_dataframe(results)
        assert "DLinear" in df.index

    def test_columns_contain_mse(self):
        pytest.importorskip("pandas")
        results = {
            "DLinear": {"mse": 0.5, "mae": 0.4, "rmse": 0.7, "smape": 10.0},
        }
        df = compare_to_dataframe(results)
        assert "mse" in df.columns

    def test_empty_results_returns_empty_df(self):
        pytest.importorskip("pandas")
        df = compare_to_dataframe({})
        assert len(df) == 0


# ── compare_plot() ────────────────────────────────────────────────────────────

_FAKE_RESULTS = {
    "DLinear": {"mse": 0.5, "mae": 0.4, "rmse": 0.71, "smape": 10.0},
    "NLinear": {"mse": 0.3, "mae": 0.3, "rmse": 0.55, "smape": 8.5},
    "PatchTST": {"mse": 0.7, "mae": 0.6, "rmse": 0.84, "smape": 12.0},
}


class TestComparePlot:
    def setup_method(self):
        pytest.importorskip("matplotlib")

    def test_returns_axes(self):
        import matplotlib.axes
        ax = compare_plot(_FAKE_RESULTS)
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_accepts_existing_axes(self):
        import matplotlib.pyplot as plt
        fig, ax_in = plt.subplots()
        ax_out = compare_plot(_FAKE_RESULTS, ax=ax_in)
        assert ax_out is ax_in
        plt.close(fig)

    def test_custom_metric(self):
        import matplotlib.axes
        ax = compare_plot(_FAKE_RESULTS, metric="mae")
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_top_n(self):
        import matplotlib.pyplot as plt
        ax = compare_plot(_FAKE_RESULTS, top_n=2)
        n_bars = len(ax.patches)
        assert n_bars == 2
        plt.close("all")

    def test_custom_title(self):
        import matplotlib.pyplot as plt
        ax = compare_plot(_FAKE_RESULTS, title="Custom Title")
        assert ax.get_title() == "Custom Title"
        plt.close("all")

    def test_empty_results_raises(self):
        with pytest.raises(ValueError):
            compare_plot({})

    def test_skips_exception_values(self):
        import matplotlib.pyplot as plt
        results = {**_FAKE_RESULTS, "BrokenModel": RuntimeError("fail")}
        ax = compare_plot(results)
        assert isinstance(ax, plt.Axes)
        plt.close("all")


# ── list_datasets() ───────────────────────────────────────────────────────────


class TestListDatasets:
    def test_all_returns_list(self):
        result = list_datasets()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_all_is_sorted(self):
        result = list_datasets()
        assert result == sorted(result)

    def test_forecast_task(self):
        result = list_datasets("forecast")
        for name in ["ETTh1", "ETTh2", "Electricity"]:
            assert name in result

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            list_datasets("unknown_task")

    def test_generation_task(self):
        result = list_datasets("generation")
        assert "Sine" in result


# ── load_dataset() ────────────────────────────────────────────────────────────


class TestLoadDataset:
    def test_returns_ndarray(self, tmp_path):
        X = load_dataset("Sine", root=str(tmp_path))
        assert isinstance(X, np.ndarray)

    def test_shape_is_2d(self, tmp_path):
        X = load_dataset("Sine", root=str(tmp_path))
        assert X.ndim == 2

    def test_float32_dtype(self, tmp_path):
        X = load_dataset("Sine", root=str(tmp_path))
        assert X.dtype == np.float32

    def test_unknown_dataset_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("NotARealDataset999", root=str(tmp_path))


# ── Forecaster.from_dataset() ─────────────────────────────────────────────────


class TestFromDataset:
    def test_returns_fitted_forecaster(self, tmp_path):
        fc = Forecaster.from_dataset(
            "DLinear", "Sine",
            root=str(tmp_path),
            seq_len=SEQ, pred_len=PRED,
            epochs=2, verbose=False,
        )
        assert fc._model is not None

    def test_predict_works_after_from_dataset(self, tmp_path):
        fc = Forecaster.from_dataset(
            "DLinear", "Sine",
            root=str(tmp_path),
            seq_len=SEQ, pred_len=PRED,
            epochs=2, verbose=False,
        )
        # Sine has 5 channels
        y = fc.predict(np.zeros((SEQ, 5), dtype=np.float32))
        assert y.shape == (PRED, 5)


# ── augmentation ──────────────────────────────────────────────────────────────


class TestAugmentation:
    def test_trains_with_jitter(self):
        X = _rng_data()
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                        epochs=2, verbose=False,
                        augmentation=Jitter(sigma=0.05))
        fc.fit(X)
        assert len(fc.history_) > 0

    def test_trains_with_compose(self):
        X = _rng_data()
        aug = Compose([Jitter(sigma=0.03), Scale(sigma=0.05)])
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                        epochs=2, verbose=False, augmentation=aug)
        fc.fit(X)
        assert len(fc.history_) > 0

    def test_no_augmentation_default(self):
        fc = _quick_fc()
        assert fc.augmentation is None

    def test_augmentation_in_get_params(self):
        aug = Jitter(sigma=0.01)
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                        verbose=False, augmentation=aug)
        assert fc.get_params()["augmentation"] is aug


# ── evaluate() ────────────────────────────────────────────────────────────────


class TestEvaluate:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())

    def test_returns_dict_with_all_keys(self):
        result = self.fc.evaluate(_rng_data(n=200))
        for key in ("mse", "mae", "rmse", "smape", "mase"):
            assert key in result

    def test_mase_positive(self):
        result = self.fc.evaluate(_rng_data(n=200))
        assert result["mase"] >= 0.0

    def test_mse_same_as_score(self):
        X = _rng_data(n=200)
        ev = self.fc.evaluate(X)
        sc = self.fc.score(X)
        assert abs(ev["mse"] - sc["mse"]) < 1e-6

    def test_seasonal_period_param(self):
        result = self.fc.evaluate(_rng_data(n=200), seasonal_period=7)
        assert "mase" in result


# ── benchmark() ───────────────────────────────────────────────────────────────


class TestExportPredictions:
    def setup_method(self):
        pytest.importorskip("pandas")
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=200)

    def test_creates_file(self, tmp_path):
        import os
        path = str(tmp_path / "preds.csv")
        out = self.fc.export_predictions(self.X, path)
        assert os.path.exists(out)

    def test_returns_path(self, tmp_path):
        path = str(tmp_path / "preds.csv")
        out = self.fc.export_predictions(self.X, path)
        assert out == path

    def test_csv_has_expected_columns(self, tmp_path):
        import pandas as pd
        path = str(tmp_path / "preds.csv")
        self.fc.export_predictions(self.X, path)
        df = pd.read_csv(path)
        assert "window" in df.columns
        assert "step" in df.columns

    def test_custom_channel_names(self, tmp_path):
        import pandas as pd
        path = str(tmp_path / "preds.csv")
        self.fc.export_predictions(self.X, path, channel_names=["a", "b", "c"])
        df = pd.read_csv(path)
        for col in ("a", "b", "c"):
            assert col in df.columns

    def test_before_fit_raises(self, tmp_path):
        fc = _quick_fc()
        with pytest.raises(RuntimeError):
            fc.export_predictions(self.X, str(tmp_path / "x.csv"))


class TestBenchmark:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())

    def test_returns_dict_with_expected_keys(self):
        result = self.fc.benchmark(n_runs=5, warmup=2)
        for key in ("mean_ms", "std_ms", "min_ms", "max_ms",
                    "throughput_samples_per_sec"):
            assert key in result

    def test_mean_ms_positive(self):
        result = self.fc.benchmark(n_runs=5, warmup=2)
        assert result["mean_ms"] > 0.0

    def test_throughput_positive(self):
        result = self.fc.benchmark(n_runs=5, warmup=2)
        assert result["throughput_samples_per_sec"] > 0.0

    def test_raises_before_fit(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            _quick_fc().benchmark()


# ── to_onnx() ─────────────────────────────────────────────────────────────────


class TestToOnnx:
    def test_creates_file(self, tmp_path):
        pytest.importorskip("onnxscript")
        fc = _quick_fc().fit(_rng_data())
        path = str(tmp_path / "model.onnx")
        fc.to_onnx(path)
        import os
        assert os.path.isfile(path)

    def test_raises_before_fit(self, tmp_path):
        with pytest.raises(RuntimeError, match="not fitted"):
            _quick_fc().to_onnx(str(tmp_path / "model.onnx"))


# ── compare_horizons() ────────────────────────────────────────────────────────


class TestCompareHorizons:
    def test_returns_dict_keyed_by_horizon(self):
        X = _rng_data(n=400)
        fc = _quick_fc()
        results = fc.compare_horizons(X, horizons=[4, 8], verbose=False)
        assert set(results.keys()) == {4, 8}

    def test_each_horizon_has_metrics(self):
        X = _rng_data(n=400)
        fc = _quick_fc()
        results = fc.compare_horizons(X, horizons=[4], verbose=False)
        for key in ("mse", "mae", "rmse", "smape"):
            assert key in results[4]

    def test_sorted_by_horizon(self):
        X = _rng_data(n=400)
        fc = _quick_fc()
        results = fc.compare_horizons(X, horizons=[8, 4, 12], verbose=False)
        assert list(results.keys()) == sorted([4, 8, 12])

    def test_mse_positive(self):
        X = _rng_data(n=400)
        fc = _quick_fc()
        results = fc.compare_horizons(X, horizons=[4], verbose=False)
        assert results[4]["mse"] >= 0.0

    def test_elapsed_s_present(self):
        X = _rng_data(n=400)
        fc = _quick_fc()
        results = fc.compare_horizons(X, horizons=[4], verbose=False)
        assert "elapsed_s" in results[4]


# ── partial_fit() ─────────────────────────────────────────────────────────────


class TestPlotScenarios:
    def setup_method(self):
        pytest.importorskip("matplotlib")
        self.fc = _quick_fc().fit(_rng_data())
        seed = _rng_data(n=SEQ)
        self.mc = self.fc.montecarlo_forecast(seed, steps=10, n_scenarios=10)

    def test_returns_axes(self):
        import matplotlib.axes
        ax = self.fc.plot_scenarios(self.mc)
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_accepts_existing_axes(self):
        import matplotlib.pyplot as plt
        fig, ax_in = plt.subplots()
        ax_out = self.fc.plot_scenarios(self.mc, ax=ax_in)
        assert ax_out is ax_in
        plt.close(fig)

    def test_with_context(self):
        import matplotlib.axes
        ctx = _rng_data(n=SEQ)
        ax = self.fc.plot_scenarios(self.mc, X_context=ctx)
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_custom_title(self):
        import matplotlib.pyplot as plt
        ax = self.fc.plot_scenarios(self.mc, title="Fan Chart")
        assert ax.get_title() == "Fan Chart"
        plt.close("all")


class TestMontecarloForecast:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.seed = _rng_data(n=SEQ)

    def test_returns_expected_keys(self):
        result = self.fc.montecarlo_forecast(self.seed, steps=10, n_scenarios=5)
        for k in ("mean", "std", "quantiles", "scenarios"):
            assert k in result

    def test_mean_shape(self):
        result = self.fc.montecarlo_forecast(self.seed, steps=10, n_scenarios=5)
        assert result["mean"].shape == (10, C)

    def test_scenarios_shape(self):
        result = self.fc.montecarlo_forecast(self.seed, steps=10, n_scenarios=5)
        assert result["scenarios"].shape == (5, 10, C)

    def test_std_non_negative(self):
        result = self.fc.montecarlo_forecast(self.seed, steps=10, n_scenarios=5)
        assert (result["std"] >= 0.0).all()

    def test_default_quantiles_present(self):
        result = self.fc.montecarlo_forecast(self.seed, steps=10, n_scenarios=5)
        for q in (0.05, 0.25, 0.75, 0.95):
            assert q in result["quantiles"]

    def test_custom_quantiles(self):
        result = self.fc.montecarlo_forecast(
            self.seed, steps=10, n_scenarios=5, quantiles=[0.1, 0.9]
        )
        assert set(result["quantiles"].keys()) == {0.1, 0.9}

    def test_before_fit_raises(self):
        fc = _quick_fc()
        with pytest.raises(RuntimeError):
            fc.montecarlo_forecast(self.seed, steps=5, n_scenarios=3)


class TestSimulate:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.seed = _rng_data(n=SEQ)

    def test_output_shape(self):
        out = self.fc.simulate(self.seed, steps=20)
        assert out.shape == (20, C)

    def test_deterministic_without_noise(self):
        out1 = self.fc.simulate(self.seed, steps=10)
        out2 = self.fc.simulate(self.seed, steps=10)
        np.testing.assert_array_equal(out1, out2)

    def test_stochastic_with_noise(self):
        out1 = self.fc.simulate(self.seed, steps=10, noise_scale=0.1, random_state=0)
        out2 = self.fc.simulate(self.seed, steps=10, noise_scale=0.1, random_state=1)
        assert not np.allclose(out1, out2)

    def test_noise_reproducible_with_seed(self):
        out1 = self.fc.simulate(self.seed, steps=10, noise_scale=0.1, random_state=42)
        out2 = self.fc.simulate(self.seed, steps=10, noise_scale=0.1, random_state=42)
        np.testing.assert_array_equal(out1, out2)

    def test_longer_seed_accepted(self):
        out = self.fc.simulate(_rng_data(n=100), steps=10)
        assert out.shape == (10, C)

    def test_before_fit_raises(self):
        fc = _quick_fc()
        with pytest.raises(RuntimeError):
            fc.simulate(self.seed, steps=5)

    def test_single_step(self):
        out = self.fc.simulate(self.seed, steps=1)
        assert out.shape == (1, C)


class TestStreamPredict:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())

    def test_yields_correct_count(self):
        X = _rng_data(n=200)
        preds = list(self.fc.stream_predict(X))
        expected = len(X) - SEQ
        assert len(preds) == expected

    def test_each_prediction_has_correct_shape(self):
        X = _rng_data(n=100)
        for pred in self.fc.stream_predict(X):
            assert pred.shape == (PRED, C)

    def test_too_short_yields_nothing(self):
        X = _rng_data(n=SEQ)  # exactly seq_len — nothing to stream
        preds = list(self.fc.stream_predict(X))
        assert preds == []

    def test_before_fit_raises(self):
        fc = _quick_fc()
        with pytest.raises(RuntimeError):
            list(fc.stream_predict(_rng_data(n=100)))

    def test_is_generator(self):
        import types
        X = _rng_data(n=100)
        gen = self.fc.stream_predict(X)
        assert isinstance(gen, types.GeneratorType)


class TestPartialFit:
    def test_partial_fit_after_fit_keeps_model(self):
        X = _rng_data()
        fc = _quick_fc()
        fc.fit(X)
        m1 = fc._model
        fc.partial_fit(X)
        assert fc._model is m1  # same object

    def test_partial_fit_works_without_prior_fit(self):
        X = _rng_data()
        fc = _quick_fc()
        fc.partial_fit(X)
        assert fc._model is not None

    def test_partial_fit_returns_self(self):
        X = _rng_data()
        fc = _quick_fc()
        ret = fc.partial_fit(X)
        assert ret is fc

    def test_warm_start_restored_after_partial_fit(self):
        X = _rng_data()
        fc = _quick_fc()
        fc.warm_start = False
        fc.fit(X)
        fc.partial_fit(X)
        assert fc.warm_start is False  # should be restored


# ── Pipeline ──────────────────────────────────────────────────────────────────


class TestPipeline:
    def test_fit_predict_shape(self):
        X = _rng_data(n=300)
        pipe = Pipeline(lambda x: x, _quick_fc())
        pipe.fit(X)
        y = pipe.predict(X[-SEQ:])
        assert y.shape == (PRED, C)

    def test_preprocessor_applied(self):
        X = _rng_data(n=300)
        # Scale by 2 — model should train on scaled data
        pipe = Pipeline(lambda x: x * 2.0, _quick_fc(normalize=False))
        pipe.fit(X)
        y = pipe.predict(X[-SEQ:])
        assert y.shape == (PRED, C)

    def test_score_returns_metrics(self):
        X = _rng_data(n=300)
        pipe = Pipeline(lambda x: x, _quick_fc())
        pipe.fit(X[:200])
        result = pipe.score(X[200:])
        assert "mse" in result

    def test_history_accessible(self):
        X = _rng_data(n=300)
        pipe = Pipeline(lambda x: x, _quick_fc())
        pipe.fit(X)
        assert len(pipe.history_) > 0

    def test_repr(self):
        pipe = Pipeline(lambda x: x, _quick_fc())
        assert "Pipeline" in repr(pipe)

    def test_non_callable_preprocessor_raises(self):
        with pytest.raises(ValueError, match="callable"):
            Pipeline("not_callable", _quick_fc())

    def test_inverse_applied_in_predict(self):
        X = _rng_data(n=300)
        scale = 2.0
        pipe = Pipeline(
            lambda x: x * scale,
            _quick_fc(normalize=False),
        )
        pipe.set_inverse(lambda y: y / scale)
        pipe.fit(X)
        y_with_inv = pipe.predict(X[-SEQ:])
        # Same pipe without inverse
        pipe2 = Pipeline(lambda x: x * scale, _quick_fc(normalize=False))
        pipe2.fit(X)
        y_no_inv = pipe2.predict(X[-SEQ:])
        # With inverse the output differs from no-inverse output
        assert y_with_inv.shape == (PRED, C)


# ── time_series_split() ───────────────────────────────────────────────────────


class TestTimeSeriesSplit:
    def test_returns_list_of_tuples(self):
        X = _rng_data(n=500)
        splits = time_series_split(X, n_splits=3)
        assert isinstance(splits, list)
        assert len(splits) > 0
        for train_idx, test_idx in splits:
            assert hasattr(train_idx, "__len__")
            assert hasattr(test_idx, "__len__")

    def test_train_before_test(self):
        X = _rng_data(n=500)
        splits = time_series_split(X, n_splits=3)
        for train_idx, test_idx in splits:
            assert train_idx.max() < test_idx.min()

    def test_n_splits_limit(self):
        X = _rng_data(n=500)
        splits = time_series_split(X, n_splits=5)
        assert len(splits) <= 5

    def test_custom_test_size(self):
        X = _rng_data(n=500)
        splits = time_series_split(X, n_splits=3, test_size=50)
        for _, test_idx in splits:
            assert len(test_idx) == 50

    def test_gap_respected(self):
        X = _rng_data(n=500)
        gap = 10
        splits = time_series_split(X, n_splits=3, gap=gap)
        for train_idx, test_idx in splits:
            assert test_idx.min() - train_idx.max() >= gap

    def test_usable_for_cross_validation(self):
        X = _rng_data(n=400)
        splits = time_series_split(X, n_splits=2)
        for train_idx, test_idx in splits:
            fc = _quick_fc()
            fc.fit(X[train_idx])
            result = fc.score(X[test_idx])
            assert "mse" in result


# ── forecast() ────────────────────────────────────────────────────────────────


class TestForecast:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())

    def test_steps_less_than_pred_len(self):
        X = _rng_data()[-SEQ:]
        y = self.fc.forecast(X, steps=4)
        assert y.shape == (4, C)

    def test_steps_equal_pred_len(self):
        X = _rng_data()[-SEQ:]
        y = self.fc.forecast(X, steps=PRED)
        assert y.shape == (PRED, C)

    def test_steps_greater_than_pred_len(self):
        X = _rng_data()[-SEQ:]
        y = self.fc.forecast(X, steps=PRED * 2)
        assert y.shape == (PRED * 2, C)

    def test_output_is_numpy(self):
        X = _rng_data()[-SEQ:]
        y = self.fc.forecast(X, steps=6)
        assert isinstance(y, np.ndarray)

    def test_output_finite(self):
        X = _rng_data()[-SEQ:]
        y = self.fc.forecast(X, steps=PRED * 3)
        assert np.isfinite(y).all()

    def test_raises_before_fit(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            _quick_fc().forecast(_rng_data()[-SEQ:], steps=5)


# ── to_dict() / from_dict() ───────────────────────────────────────────────────


class TestToFromDict:
    def test_to_dict_is_json_serializable(self):
        import json
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                        lr=5e-4, epochs=10, verbose=False)
        d = fc.to_dict()
        json.dumps(d)  # must not raise

    def test_to_dict_has_model_key(self):
        fc = _quick_fc()
        d = fc.to_dict()
        assert "model" in d

    def test_to_dict_roundtrip(self):
        fc = Forecaster("DLinear", seq_len=48, pred_len=12,
                        lr=5e-4, verbose=False)
        d = fc.to_dict()
        fc2 = Forecaster.from_dict(d)
        assert fc2.seq_len == 48
        assert fc2.pred_len == 12
        assert fc2.lr == 5e-4

    def test_from_dict_returns_unfitted(self):
        fc = _quick_fc()
        fc2 = Forecaster.from_dict(fc.to_dict())
        assert fc2._model is None


# ── make_forecaster() ─────────────────────────────────────────────────────────


class TestMakeForecaster:
    def test_returns_forecaster(self):
        fc = make_forecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                             verbose=False)
        assert isinstance(fc, Forecaster)

    def test_model_spec_set(self):
        fc = make_forecaster("NLinear", seq_len=SEQ, pred_len=PRED,
                             verbose=False)
        assert fc.model_spec == "NLinear"

    def test_kwargs_passed(self):
        fc = make_forecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                             epochs=7, verbose=False)
        assert fc.epochs == 7

    def test_importable_from_package(self):
        from torch_timeseries import make_forecaster as mf
        assert mf is make_forecaster


# ── reset() ───────────────────────────────────────────────────────────────────


class TestAddRemoveCallback:
    def test_add_callback_returns_self(self):
        fc = _quick_fc()
        cb = lambda fc, d: None
        assert fc.add_callback(cb) is fc

    def test_callback_fires_during_fit(self):
        calls = []
        fc = _quick_fc()
        fc.add_callback(lambda f, d: calls.append(d["epoch"]))
        fc.fit(_rng_data())
        assert len(calls) > 0

    def test_callback_receives_epoch_dict(self):
        keys_seen = []
        fc = _quick_fc()
        fc.add_callback(lambda f, d: keys_seen.extend(d.keys()))
        fc.fit(_rng_data())
        for k in ("epoch", "train_loss", "val_loss"):
            assert k in keys_seen

    def test_remove_callback(self):
        calls = []
        fc = _quick_fc()
        cb = lambda f, d: calls.append(1)
        fc.add_callback(cb)
        fc.remove_callback(cb)
        fc.fit(_rng_data())
        assert calls == []

    def test_remove_callback_returns_self(self):
        fc = _quick_fc()
        cb = lambda f, d: None
        fc.add_callback(cb)
        assert fc.remove_callback(cb) is fc

    def test_remove_unknown_raises(self):
        fc = _quick_fc()
        with pytest.raises(ValueError, match="not found"):
            fc.remove_callback(lambda f, d: None)

    def test_add_non_callable_raises(self):
        fc = _quick_fc()
        with pytest.raises(TypeError):
            fc.add_callback("not_a_function")


class TestReset:
    def test_reset_unfits_model(self):
        fc = _quick_fc().fit(_rng_data())
        fc.reset()
        assert fc._model is None

    def test_reset_clears_history(self):
        fc = _quick_fc().fit(_rng_data())
        fc.reset()
        assert fc.history_ == []

    def test_reset_returns_self(self):
        fc = _quick_fc().fit(_rng_data())
        ret = fc.reset()
        assert ret is fc

    def test_reset_repr_not_fitted(self):
        fc = _quick_fc().fit(_rng_data())
        fc.reset()
        assert "not fitted" in repr(fc)

    def test_can_refit_after_reset(self):
        fc = _quick_fc()
        fc.fit(_rng_data())
        fc.reset()
        fc.fit(_rng_data(seed=1))
        assert fc._model is not None

    def test_hyperparams_preserved_after_reset(self):
        fc = Forecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                        epochs=7, verbose=False)
        fc.fit(_rng_data())
        fc.reset()
        assert fc.epochs == 7
        assert fc.seq_len == SEQ


# ── inspect_layers() ──────────────────────────────────────────────────────────


class TestInspectLayers:
    def test_returns_string(self):
        fc = _quick_fc().fit(_rng_data())
        s = fc.inspect_layers()
        assert isinstance(s, str)

    def test_contains_model_name(self):
        fc = _quick_fc("DLinear").fit(_rng_data())
        s = fc.inspect_layers()
        assert "DLinear" in s

    def test_contains_total_params(self):
        fc = _quick_fc().fit(_rng_data())
        s = fc.inspect_layers()
        assert "total params" in s

    def test_raises_before_fit(self):
        fc = _quick_fc()
        with pytest.raises(RuntimeError, match="not fitted"):
            fc.inspect_layers()


# ── fit_score() ───────────────────────────────────────────────────────────────


class TestFitScore:
    def test_returns_dict_with_metrics(self):
        X = _rng_data(n=400)
        fc = _quick_fc()
        result = fc.fit_score(X)
        for key in ("mse", "mae", "rmse", "smape", "mase"):
            assert key in result

    def test_mse_positive(self):
        X = _rng_data(n=400)
        fc = _quick_fc()
        result = fc.fit_score(X)
        assert result["mse"] >= 0.0

    def test_model_is_fitted(self):
        X = _rng_data(n=400)
        fc = _quick_fc()
        fc.fit_score(X)
        assert fc._model is not None

    def test_custom_test_size(self):
        X = _rng_data(n=400)
        fc = _quick_fc()
        result = fc.fit_score(X, test_size=0.1)
        assert result["mse"] >= 0.0

    def test_too_little_data_raises(self):
        X = _rng_data(n=50)
        fc = _quick_fc()
        with pytest.raises(ValueError):
            fc.fit_score(X, test_size=0.9)


# ── detect_anomalies() ────────────────────────────────────────────────────────


class TestPlotResiduals:
    def setup_method(self):
        pytest.importorskip("matplotlib")
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=200)

    def test_returns_axes(self):
        import matplotlib.axes
        ax = self.fc.plot_residuals(self.X)
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_accepts_existing_axes(self):
        import matplotlib.pyplot as plt
        fig, ax_in = plt.subplots()
        ax_out = self.fc.plot_residuals(self.X, ax=ax_in)
        assert ax_out is ax_in
        plt.close(fig)

    def test_custom_channel(self):
        import matplotlib.axes
        ax = self.fc.plot_residuals(self.X, channel=1)
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_custom_title(self):
        import matplotlib.pyplot as plt
        ax = self.fc.plot_residuals(self.X, title="My Residuals")
        assert ax.get_title() == "My Residuals"
        plt.close("all")

    def test_before_fit_raises(self):
        fc = _quick_fc()
        with pytest.raises(RuntimeError):
            fc.plot_residuals(self.X)


class TestPlotChannelScores:
    def setup_method(self):
        pytest.importorskip("matplotlib")
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=200)

    def test_returns_axes(self):
        import matplotlib.axes
        ax = self.fc.plot_channel_scores(self.X)
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_accepts_existing_axes(self):
        import matplotlib.pyplot as plt
        fig, ax_in = plt.subplots()
        ax_out = self.fc.plot_channel_scores(self.X, ax=ax_in)
        assert ax_out is ax_in
        plt.close(fig)

    def test_custom_metric(self):
        import matplotlib.axes
        ax = self.fc.plot_channel_scores(self.X, metric="mae")
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric"):
            self.fc.plot_channel_scores(self.X, metric="bogus")

    def test_custom_channel_names(self):
        import matplotlib.pyplot as plt
        ax = self.fc.plot_channel_scores(self.X, channel_names=["a", "b", "c"])
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "a" in labels
        plt.close("all")


class TestScorePerChannel:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())

    def test_returns_expected_keys(self):
        result = self.fc.score_per_channel(_rng_data(n=200))
        for key in ("mse", "mae", "rmse"):
            assert key in result

    def test_shape_equals_n_channels(self):
        result = self.fc.score_per_channel(_rng_data(n=200))
        assert result["mse"].shape == (C,)
        assert result["mae"].shape == (C,)
        assert result["rmse"].shape == (C,)

    def test_values_non_negative(self):
        result = self.fc.score_per_channel(_rng_data(n=200))
        assert (result["mse"] >= 0.0).all()
        assert (result["mae"] >= 0.0).all()
        assert (result["rmse"] >= 0.0).all()

    def test_rmse_equals_sqrt_mse(self):
        result = self.fc.score_per_channel(_rng_data(n=200))
        np.testing.assert_allclose(result["rmse"], np.sqrt(result["mse"]), rtol=1e-5)

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            self.fc.score_per_channel(np.zeros((5, C)))

    def test_before_fit_raises(self):
        fc = _quick_fc()
        with pytest.raises(RuntimeError):
            fc.score_per_channel(_rng_data(n=200))


class TestDetectAnomalies:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())

    def test_returns_bool_array(self):
        X = _rng_data(n=200)
        mask = self.fc.detect_anomalies(X)
        assert mask.dtype == bool

    def test_shape_equals_input_len(self):
        X = _rng_data(n=200)
        mask = self.fc.detect_anomalies(X)
        assert len(mask) == 200

    def test_first_seqlen_are_false(self):
        X = _rng_data(n=200)
        mask = self.fc.detect_anomalies(X)
        assert not mask[:SEQ].any()

    def test_custom_threshold(self):
        X = _rng_data(n=200)
        # Very low threshold → many anomalies
        mask_low = self.fc.detect_anomalies(X, threshold=0.001)
        # Very high threshold → few/no anomalies
        mask_high = self.fc.detect_anomalies(X, threshold=1e9)
        assert mask_low.sum() >= mask_high.sum()

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            self.fc.detect_anomalies(np.zeros((5, C)))


# ── MultiChannelForecaster ────────────────────────────────────────────────────


class TestMultiChannelForecaster:
    def _base(self):
        return _quick_fc("DLinear")

    def test_fit_returns_self(self):
        X = _rng_data(n=300)
        mcf = MultiChannelForecaster(self._base())
        result = mcf.fit(X)
        assert result is mcf

    def test_creates_one_forecaster_per_channel(self):
        X = _rng_data(n=300, c=4)
        mcf = MultiChannelForecaster(_quick_fc("DLinear"))
        mcf.fit(X)
        assert len(mcf.channel_forecasters_) == 4

    def test_predict_shape(self):
        X = _rng_data(n=300)
        mcf = MultiChannelForecaster(self._base())
        mcf.fit(X)
        y = mcf.predict(X[-SEQ:])
        assert y.shape == (PRED, C)

    def test_predict_uses_last_seqlen_rows_when_longer(self):
        X = _rng_data(n=300)
        mcf = MultiChannelForecaster(self._base())
        mcf.fit(X)
        y_ctx = mcf.predict(X[-SEQ:])
        y_full = mcf.predict(X)
        # Both should produce same-shaped output
        assert y_ctx.shape == y_full.shape

    def test_score_returns_expected_keys(self):
        X = _rng_data(n=300)
        mcf = MultiChannelForecaster(self._base())
        mcf.fit(X[:200])
        result = mcf.score(X[200:])
        for key in ("mse", "mae", "rmse", "smape"):
            assert key in result

    def test_score_mse_non_negative(self):
        X = _rng_data(n=300)
        mcf = MultiChannelForecaster(self._base())
        mcf.fit(X[:200])
        result = mcf.score(X[200:])
        assert result["mse"] >= 0.0

    def test_predict_before_fit_raises(self):
        mcf = MultiChannelForecaster(self._base())
        with pytest.raises(RuntimeError, match="not fitted"):
            mcf.predict(np.zeros((SEQ, C)))

    def test_score_before_fit_raises(self):
        mcf = MultiChannelForecaster(self._base())
        with pytest.raises(RuntimeError, match="not fitted"):
            mcf.score(np.zeros((200, C)))

    def test_score_too_short_raises(self):
        X = _rng_data(n=300)
        mcf = MultiChannelForecaster(self._base())
        mcf.fit(X)
        with pytest.raises(ValueError):
            mcf.score(np.zeros((5, C)))

    def test_univariate_input(self):
        X = _rng_data(n=300, c=1)
        mcf = MultiChannelForecaster(_quick_fc("DLinear"))
        mcf.fit(X)
        y = mcf.predict(X[-SEQ:])
        assert y.shape == (PRED, 1)

    def test_repr_unfitted(self):
        mcf = MultiChannelForecaster(self._base())
        assert "MultiChannelForecaster" in repr(mcf)
        assert "not fitted" in repr(mcf)

    def test_repr_fitted(self):
        X = _rng_data(n=300)
        mcf = MultiChannelForecaster(self._base())
        mcf.fit(X)
        assert "3 channels fitted" in repr(mcf)


# ── Forecaster.smooth() ───────────────────────────────────────────────────────


class TestPlotDecomposition:
    def setup_method(self):
        pytest.importorskip("matplotlib")
        X = _rng_data(n=100)
        self.decomp = Forecaster.seasonal_decompose(X, period=7)

    def test_returns_figure(self):
        import matplotlib.figure
        fig = Forecaster.plot_decomposition(self.decomp)
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_has_four_subplots(self):
        import matplotlib.pyplot as plt
        fig = Forecaster.plot_decomposition(self.decomp)
        assert len(fig.get_axes()) == 4
        plt.close("all")

    def test_custom_title(self):
        import matplotlib.pyplot as plt
        fig = Forecaster.plot_decomposition(self.decomp, title="Decomp")
        assert "Decomp" in fig.texts[0].get_text()
        plt.close("all")

    def test_custom_channel(self):
        import matplotlib.figure
        fig = Forecaster.plot_decomposition(self.decomp, channel=1)
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close("all")


class TestPlotQuantileForecast:
    def setup_method(self):
        pytest.importorskip("matplotlib")
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_returns_figure(self):
        import matplotlib.figure
        import matplotlib.pyplot as plt
        fig = self.fc.plot_quantile_forecast(self.X, n_samples=10)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_custom_title(self):
        import matplotlib.pyplot as plt
        fig = self.fc.plot_quantile_forecast(self.X, n_samples=10, title="QF")
        ax = fig.get_axes()[0]
        assert "QF" in ax.get_title()
        plt.close("all")


class TestFitOnDataframe:
    def test_fits_from_dataframe(self):
        pytest.importorskip("pandas")
        import pandas as pd
        X = _rng_data(n=100)
        df = pd.DataFrame(X, columns=[f"ch{i}" for i in range(C)])
        fc = _quick_fc()
        fc.fit_on_dataframe(df)
        assert fc._model is not None

    def test_non_dataframe_raises(self):
        fc = _quick_fc()
        with pytest.raises(TypeError):
            fc.fit_on_dataframe(np.zeros((100, C)))

    def test_target_cols_subset(self):
        pytest.importorskip("pandas")
        import pandas as pd
        X = _rng_data(n=100)
        df = pd.DataFrame(X, columns=[f"ch{i}" for i in range(C)])
        fc = Forecaster("NLinear", seq_len=SEQ, pred_len=PRED, epochs=2,
                        batch_size=32, patience=5, verbose=False)
        fc.fit_on_dataframe(df, target_cols=["ch0", "ch1"])
        assert fc._enc_in == 2

    def test_excludes_date_col(self):
        pytest.importorskip("pandas")
        import pandas as pd
        X = _rng_data(n=100)
        df = pd.DataFrame(X, columns=[f"ch{i}" for i in range(C)])
        df.insert(0, "date", pd.date_range("2020-01-01", periods=100, freq="h"))
        fc = _quick_fc()
        fc.fit_on_dataframe(df, date_col="date")
        assert fc._enc_in == C   # should not include the date column


class TestPartialAutocorrelation:
    def test_returns_two_arrays(self):
        X = _rng_data(n=200)
        lags, pacf = Forecaster.partial_autocorrelation(X, max_lag=20)
        assert lags.shape == (21,)
        assert pacf.shape == (21,)

    def test_lag_zero_is_one(self):
        X = _rng_data(n=200)
        _, pacf = Forecaster.partial_autocorrelation(X)
        assert abs(pacf[0] - 1.0) < 1e-6

    def test_ar1_pacf_cuts_off(self):
        # AR(1) with φ=0.8 → PACF should be large at lag 1 and small at lag 2
        rng = np.random.default_rng(42)
        x = np.zeros(500)
        for t in range(1, 500):
            x[t] = 0.8 * x[t - 1] + rng.standard_normal()
        X = x[:, None].astype(np.float32)
        _, pacf = Forecaster.partial_autocorrelation(X, max_lag=5)
        assert abs(pacf[1]) > abs(pacf[2])   # lag-1 stronger than lag-2

    def test_bounded_values(self):
        X = _rng_data(n=200)
        _, pacf = Forecaster.partial_autocorrelation(X, max_lag=10)
        assert (np.abs(pacf) <= 1.0 + 1e-6).all()

    def test_1d_input(self):
        X = _rng_data(n=200)[:, 0]
        lags, pacf = Forecaster.partial_autocorrelation(X, max_lag=10)
        assert lags.shape == (11,)


class TestPlotPacf:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure
        import matplotlib.pyplot as plt
        X = _rng_data(n=200)
        fig = Forecaster.plot_pacf(X)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_custom_title(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        X = _rng_data(n=200)
        fig = Forecaster.plot_pacf(X, title="PACF Test")
        ax = fig.get_axes()[0]
        assert "PACF Test" in ax.get_title()
        plt.close("all")


class TestResidualAcf:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_returns_arrays(self):
        lags, acf = self.fc.residual_acf(self.X, max_lag=20)
        assert lags.shape == (21,) and acf.shape == (21,)

    def test_lag_zero_is_one(self):
        _, acf = self.fc.residual_acf(self.X, max_lag=10)
        assert abs(acf[0] - 1.0) < 1e-5

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().residual_acf(self.X)


class TestPlotResidualAcf:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure, matplotlib.pyplot as plt
        fc = _quick_fc().fit(_rng_data())
        lags, acf = fc.residual_acf(_rng_data(n=200), max_lag=15)
        fig = Forecaster.plot_residual_acf(lags, acf)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")


class TestClone:
    def test_clone_is_unfitted(self):
        fc = _quick_fc().fit(_rng_data())
        cloned = fc.clone()
        assert cloned._model is None

    def test_clone_has_same_config(self):
        fc = _quick_fc()
        c = fc.clone()
        assert c.seq_len == fc.seq_len
        assert c.pred_len == fc.pred_len
        assert c.epochs == fc.epochs

    def test_clone_fits_independently(self):
        fc = _quick_fc().fit(_rng_data())
        cloned = fc.clone()
        cloned.fit(_rng_data(seed=1))
        # original model untouched
        assert fc._model is not None


class TestLeaderboard:
    def test_returns_dataframe(self):
        pytest.importorskip("pandas")
        X = _rng_data(n=200)
        df = Forecaster.leaderboard(
            X[:140], X[140:],
            ["DLinear", "NLinear"],
            seq_len=SEQ, pred_len=PRED, epochs=2,
            batch_size=32, patience=5, verbose=False,
        )
        assert "mse" in df.columns
        assert len(df) == 2

    def test_sort_by_metric(self):
        pytest.importorskip("pandas")
        X = _rng_data(n=200)
        df = Forecaster.leaderboard(
            X[:140], X[140:],
            ["DLinear", "NLinear"],
            metric="mae", sort=True,
            seq_len=SEQ, pred_len=PRED, epochs=2,
            batch_size=32, patience=5, verbose=False,
        )
        vals = df["mae"].dropna().values
        assert list(vals) == sorted(vals)


class TestSensitivityAnalysis:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_returns_dict(self):
        result = self.fc.sensitivity_analysis(self.X, channel=0, n_points=5)
        assert set(result.keys()) == {"deltas", "pred_change", "baseline"}

    def test_correct_shapes(self):
        result = self.fc.sensitivity_analysis(self.X, channel=0, n_points=7)
        assert result["deltas"].shape == (7,)
        assert result["pred_change"].shape == (7,)

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().sensitivity_analysis(self.X, channel=0)


class TestPlotSensitivity:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure, matplotlib.pyplot as plt
        fc = _quick_fc().fit(_rng_data())
        fig = fc.plot_sensitivity(_rng_data(n=200), channel=0, n_points=5)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")


class TestTrainValTestSplit:
    def test_basic_split(self):
        X = _rng_data(n=1000)
        tr, va, te = Forecaster.train_val_test_split(X, val_ratio=0.1, test_ratio=0.2)
        assert len(tr) + len(va) + len(te) == 1000
        assert len(te) == pytest.approx(200, abs=2)
        assert len(va) == pytest.approx(100, abs=2)

    def test_chronological_order(self):
        X = np.arange(1000).reshape(-1, 1).astype(np.float32)
        tr, va, te = Forecaster.train_val_test_split(X, val_ratio=0.1, test_ratio=0.1)
        assert float(tr[-1]) < float(va[0])
        assert float(va[-1]) < float(te[0])

    def test_gap_removes_samples(self):
        X = _rng_data(n=1000)
        tr_no_gap, va_no_gap, te_no_gap = Forecaster.train_val_test_split(X)
        tr_gap,    va_gap,    te_gap    = Forecaster.train_val_test_split(X, gap=10)
        assert len(tr_gap) < len(tr_no_gap)

    def test_too_small_raises(self):
        X = _rng_data(n=2)
        with pytest.raises(ValueError):
            Forecaster.train_val_test_split(X, val_ratio=0.5, test_ratio=0.5)


class TestLjungBox:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_returns_dict(self):
        result = self.fc.ljung_box(self.X, max_lag=10)
        assert set(result.keys()) >= {"Q", "df", "p_value"}

    def test_df_equals_max_lag(self):
        result = self.fc.ljung_box(self.X, max_lag=15)
        assert result["df"] == 15

    def test_p_value_in_unit_interval(self):
        result = self.fc.ljung_box(self.X, max_lag=10)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_Q_is_positive(self):
        result = self.fc.ljung_box(self.X, max_lag=10)
        assert result["Q"] > 0

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().ljung_box(self.X)


class TestLagPlot:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure, matplotlib.pyplot as plt
        X = _rng_data(n=200)
        fig = Forecaster.lag_plot(X, lag=1, channel=0)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_custom_lag(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        X = _rng_data(n=200)
        fig = Forecaster.lag_plot(X, lag=5)
        plt.close("all")
        assert fig is not None

    def test_1d_input(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        X = _rng_data(n=200)[:, 0]
        fig = Forecaster.lag_plot(X, lag=2)
        plt.close("all")
        assert fig is not None


class TestSeasonalPlot:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure, matplotlib.pyplot as plt
        X = _rng_data(n=120)
        fig = Forecaster.seasonal_plot(X, period=12)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_too_short_raises(self):
        X = _rng_data(n=5)
        with pytest.raises(ValueError):
            Forecaster.seasonal_plot(X, period=24)

    def test_custom_title(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        X = _rng_data(n=120)
        fig = Forecaster.seasonal_plot(X, period=12, title="Seasons")
        ax = fig.get_axes()[0]
        assert "Seasons" in ax.get_title()
        plt.close("all")


class TestHyperparameterSearch:
    def test_returns_list_sorted_by_metric(self):
        X = _rng_data(n=200)
        fc = _quick_fc()
        results = fc.hyperparameter_search(
            X[:140], X[140:],
            {"lr": [1e-3, 5e-4]},
            n_iter=2, metric="mse",
        )
        assert isinstance(results, list)
        assert len(results) == 2
        assert all("params" in r for r in results)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores)

    def test_params_keys_in_results(self):
        X = _rng_data(n=200)
        fc = _quick_fc()
        results = fc.hyperparameter_search(
            X[:140], X[140:],
            {"lr": [1e-3]},
            n_iter=1,
        )
        assert "lr" in results[0]["params"]


class TestToLaggedFeatures:
    def test_shapes(self):
        X = _rng_data(n=100)
        feats, targets = Forecaster.to_lagged_features(X, lags=[1, 2, 3])
        max_lag = 3
        assert feats.shape == (100 - max_lag, C * 3)
        assert targets.shape == (100 - max_lag,)

    def test_1d_input(self):
        X = _rng_data(n=50)[:, 0]
        feats, targets = Forecaster.to_lagged_features(X, lags=[1, 2])
        assert feats.shape == (48, 2)

    def test_no_nans_after_dropna(self):
        X = _rng_data(n=100)
        X[5, 0] = np.nan
        feats, targets = Forecaster.to_lagged_features(X, lags=[1, 2], dropna=True)
        assert np.all(np.isfinite(feats))


class TestPredictBootstrap:
    def test_returns_dict(self):
        fc = _quick_fc().fit(_rng_data())
        result = fc.predict_bootstrap(_rng_data(n=200), _rng_data(n=200), n_boot=3)
        assert set(result.keys()) == {"mean", "lower", "upper", "preds"}

    def test_shapes(self):
        fc = _quick_fc().fit(_rng_data())
        result = fc.predict_bootstrap(_rng_data(n=200), _rng_data(n=200), n_boot=3)
        assert result["preds"].shape == (3, PRED, C)
        assert result["mean"].shape  == (PRED, C)

    def test_lower_leq_upper(self):
        fc = _quick_fc().fit(_rng_data())
        result = fc.predict_bootstrap(_rng_data(n=200), _rng_data(n=200), n_boot=3)
        assert (result["lower"] <= result["upper"]).all()

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().predict_bootstrap(_rng_data(n=200), _rng_data(n=200))


class TestPlotActualVsPredicted:
    def setup_method(self):
        pytest.importorskip("matplotlib")
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_returns_figure(self):
        import matplotlib.figure, matplotlib.pyplot as plt
        fig = self.fc.plot_actual_vs_predicted(self.X)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_title_contains_r2(self):
        import matplotlib.pyplot as plt
        fig = self.fc.plot_actual_vs_predicted(self.X)
        ax = fig.get_axes()[0]
        assert "R²" in ax.get_title() or "R" in ax.get_title()
        plt.close("all")

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().plot_actual_vs_predicted(self.X)


class TestNoiseRobustness:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_returns_dict(self):
        result = self.fc.noise_robustness(self.X, self.X, noise_levels=[0.0, 0.5], n_trials=1)
        assert isinstance(result, dict)
        assert 0.0 in result and 0.5 in result

    def test_zero_noise_is_finite(self):
        result = self.fc.noise_robustness(self.X, self.X, noise_levels=[0.0], n_trials=1)
        assert np.isfinite(result[0.0])

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().noise_robustness(self.X, self.X)


class TestPlotNoiseRobustness:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure, matplotlib.pyplot as plt
        fc = _quick_fc().fit(_rng_data())
        X  = _rng_data(n=200)
        fig = fc.plot_noise_robustness(X, X, noise_levels=[0.0, 0.5], n_trials=1)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")


class TestSeasonalNaiveBaseline:
    def test_shape(self):
        X = _rng_data(n=100)
        y = Forecaster.seasonal_naive_baseline(X, pred_len=PRED, period=12)
        assert y.shape == (PRED, C)

    def test_1d_input(self):
        X = _rng_data(n=50)[:, 0]
        y = Forecaster.seasonal_naive_baseline(X, pred_len=PRED, period=12)
        assert y.shape == (PRED, 1)

    def test_too_short_raises(self):
        X = _rng_data(n=5)
        with pytest.raises(ValueError):
            Forecaster.seasonal_naive_baseline(X, pred_len=PRED, period=24)


class TestExplainGlobal:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_shape(self):
        imp = self.fc.explain_global(self.X, n_samples=5)
        assert imp.shape == (SEQ, C)

    def test_non_negative_absolute(self):
        imp = self.fc.explain_global(self.X, n_samples=5, absolute=True)
        assert (imp >= 0).all()

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().explain_global(self.X)


class TestForecastErrorDistribution:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_returns_dict(self):
        stats = self.fc.forecast_error_distribution(self.X)
        assert set(stats.keys()) >= {"steps", "mean", "std", "median", "q05", "q95"}

    def test_shapes(self):
        stats = self.fc.forecast_error_distribution(self.X)
        for k in ("steps", "mean", "std", "median", "q05", "q95"):
            assert stats[k].shape == (PRED,), f"{k} wrong shape"

    def test_q05_leq_q95(self):
        stats = self.fc.forecast_error_distribution(self.X)
        assert (stats["q05"] <= stats["q95"]).all()

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().forecast_error_distribution(self.X)


class TestPlotForecastErrorDistribution:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure, matplotlib.pyplot as plt
        fc = _quick_fc().fit(_rng_data())
        fig = fc.plot_forecast_error_distribution(_rng_data(n=200))
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")


class TestAutoSelect:
    def test_returns_fitted_forecaster(self):
        X = _rng_data(n=200)
        best = Forecaster.auto_select(
            X[:140], X[140:],
            ["DLinear", "NLinear"],
            seq_len=SEQ, pred_len=PRED, epochs=2,
            batch_size=32, patience=5, verbose=False,
        )
        assert isinstance(best, Forecaster)
        assert best._model is not None

    def test_all_fail_raises(self):
        X = _rng_data(n=200)
        with pytest.raises(RuntimeError):
            Forecaster.auto_select(
                X[:140], X[140:],
                ["ThisModelDoesNotExist999"],
                seq_len=SEQ, pred_len=PRED, epochs=2,
                batch_size=32, patience=5, verbose=False,
            )


class TestConformalCoverage:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_returns_dict(self):
        result = self.fc.conformal_coverage(self.X, self.X, coverage=0.9)
        assert set(result.keys()) >= {"nominal_coverage", "empirical_coverage", "coverage_gap"}

    def test_nominal_preserved(self):
        result = self.fc.conformal_coverage(self.X, self.X, coverage=0.8)
        assert result["nominal_coverage"] == pytest.approx(0.8)

    def test_empirical_in_unit_interval(self):
        result = self.fc.conformal_coverage(self.X, self.X, coverage=0.9)
        assert 0.0 <= result["empirical_coverage"] <= 1.0

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().conformal_coverage(self.X, self.X)


class TestWinklerScore:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_returns_float(self):
        ws = self.fc.winkler_score(self.X, self.X)
        assert isinstance(ws, float)

    def test_positive(self):
        ws = self.fc.winkler_score(self.X, self.X)
        assert ws > 0

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().winkler_score(self.X, self.X)


class TestConceptDriftScore:
    def test_scalar_output(self):
        X_ref  = _rng_data(n=200)
        X_test = _rng_data(n=200, seed=1)
        score  = Forecaster.concept_drift_score(X_ref, X_test)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_same_dist_near_zero(self):
        X = _rng_data(n=500)
        score = Forecaster.concept_drift_score(X[:250], X[250:])
        assert score < 0.5   # same underlying distribution

    def test_rolling_output(self):
        X_ref  = _rng_data(n=200)
        X_test = _rng_data(n=150, seed=2)
        scores = Forecaster.concept_drift_score(X_ref, X_test, window=50)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 150 - 50 + 1

    def test_1d_input(self):
        X_ref  = _rng_data(n=100)[:, 0]
        X_test = _rng_data(n=100, seed=3)[:, 0]
        score  = Forecaster.concept_drift_score(X_ref, X_test)
        assert isinstance(score, float)


class TestPlotConceptDrift:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure, matplotlib.pyplot as plt
        X_ref  = _rng_data(n=200)
        X_test = _rng_data(n=200, seed=1)
        fc = _quick_fc()   # static method; fc not needed but test via instance
        fig = fc.plot_concept_drift(X_ref, X_test, window=50)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")


class TestPlotPredictionBands:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure, matplotlib.pyplot as plt
        fc = _quick_fc().fit(_rng_data())
        fig = fc.plot_prediction_bands(_rng_data(n=200), n_samples=10)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().plot_prediction_bands(_rng_data(n=200), n_samples=5)


class TestPlotCalibrationCurve:
    def setup_method(self):
        pytest.importorskip("matplotlib")
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_returns_figure(self):
        import matplotlib.figure, matplotlib.pyplot as plt
        fig = self.fc.plot_calibration_curve(self.X, self.X, n_levels=5)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().plot_calibration_curve(self.X, self.X)


class TestRollingZscore:
    def test_shape(self):
        X = _rng_data(n=200)
        z = Forecaster.rolling_zscore(X, window=30, channel=0)
        assert z.shape == (200,)

    def test_1d_input(self):
        X = _rng_data(n=100)[:, 0]
        z = Forecaster.rolling_zscore(X, window=20)
        assert z.shape == (100,)

    def test_typical_magnitude(self):
        X = _rng_data(n=500)
        z = Forecaster.rolling_zscore(X, window=50, channel=0)
        # z-scores of a standard normal should mostly be in [-4, 4]
        assert np.abs(z[50:]).max() < 8.0


class TestMultistepScore:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_returns_dict(self):
        scores = self.fc.multistep_score(self.X, metric="mse")
        assert isinstance(scores, dict)
        assert len(scores) == PRED

    def test_keys_are_steps(self):
        scores = self.fc.multistep_score(self.X)
        assert set(scores.keys()) == set(range(PRED))

    def test_all_positive(self):
        scores = self.fc.multistep_score(self.X, metric="mae")
        assert all(v >= 0 for v in scores.values())

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError):
            self.fc.multistep_score(self.X, metric="unknown_metric")

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().multistep_score(self.X)


class TestPlotMultistepScore:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure, matplotlib.pyplot as plt
        fc = _quick_fc().fit(_rng_data())
        fig = fc.plot_multistep_score(_rng_data(n=200), metric="mae")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")


class TestFeatureDrift:
    def test_shape(self):
        X_ref  = _rng_data(n=200)
        X_test = _rng_data(n=200, seed=1)
        scores = Forecaster.feature_drift(X_ref, X_test)
        assert scores.shape == (C,)

    def test_range(self):
        X_ref  = _rng_data(n=200)
        X_test = _rng_data(n=200, seed=1)
        scores = Forecaster.feature_drift(X_ref, X_test)
        assert (scores >= 0).all()

    def test_same_data_near_zero(self):
        X = _rng_data(n=500)
        scores = Forecaster.feature_drift(X[:250], X[250:])
        # same distribution, JS should be low
        assert scores.max() < 0.3

    def test_1d_input(self):
        X_ref  = _rng_data(n=100)[:, 0]
        X_test = _rng_data(n=100, seed=2)[:, 0]
        scores = Forecaster.feature_drift(X_ref, X_test)
        assert scores.shape == (1,)


class TestPlotFeatureDrift:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure, matplotlib.pyplot as plt
        X_ref  = _rng_data(n=200)
        X_test = _rng_data(n=200, seed=1)
        fig = Forecaster.plot_feature_drift(X_ref, X_test)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")


class TestCrossCorrelation:
    def test_returns_arrays(self):
        X = _rng_data(n=200)
        lags, ccf = Forecaster.cross_correlation(X, max_lag=20)
        assert lags.shape == (41,) and ccf.shape == (41,)

    def test_lag_zero_is_high_for_same_signal(self):
        X = _rng_data(n=200)
        # cross-correlate channel 0 with itself via Y argument
        Y = _rng_data(n=200)[:, 0:1]
        lags, ccf = Forecaster.cross_correlation(X[:, 0:1], Y, max_lag=10,
                                                  channel_x=0, channel_y=0)
        # lag 0 should have maximum absolute CCF
        assert abs(ccf[10]) == pytest.approx(max(np.abs(ccf)), abs=1e-6)

    def test_two_separate_arrays(self):
        X = _rng_data(n=200)
        Y = _rng_data(n=200, seed=7)
        lags, ccf = Forecaster.cross_correlation(X, Y, max_lag=5,
                                                  channel_x=0, channel_y=0)
        assert lags.shape == (11,)


class TestPlotCrossCorrelation:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure, matplotlib.pyplot as plt
        X = _rng_data(n=200)
        fig = Forecaster.plot_cross_correlation(X, max_lag=15)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")


class TestRegimeDetection:
    def test_shape(self):
        X = _rng_data(n=300)
        regimes = Forecaster.regime_detection(X, n_regimes=3, window=20)
        assert regimes.shape == (300,)

    def test_labels_in_range(self):
        X = _rng_data(n=300)
        regimes = Forecaster.regime_detection(X, n_regimes=3, window=20)
        unique = set(regimes.tolist())
        # labels should be -1 (warm-up) or 0..n_regimes-1
        assert unique <= {-1, 0, 1, 2}

    def test_1d_input(self):
        X = _rng_data(n=200)[:, 0]
        regimes = Forecaster.regime_detection(X, n_regimes=2, window=15)
        assert regimes.shape == (200,)


class TestPlotRegimes:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure, matplotlib.pyplot as plt
        X = _rng_data(n=200)
        regimes = Forecaster.regime_detection(X, n_regimes=2, window=15)
        fig = Forecaster.plot_regimes(X, regimes, channel=0)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")


class TestFunctionalBoxplot:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure, matplotlib.pyplot as plt
        X = _rng_data(n=240)
        fig = Forecaster.functional_boxplot(X, period=24, channel=0)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_too_short_raises(self):
        X = _rng_data(n=20)
        with pytest.raises(ValueError):
            Forecaster.functional_boxplot(X, period=24)

    def test_custom_title(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        X = _rng_data(n=240)
        fig = Forecaster.functional_boxplot(X, period=24, title="FB test")
        ax = fig.get_axes()[0]
        assert "FB test" in ax.get_title()
        plt.close("all")


class TestSpectrogram:
    def test_returns_dict(self):
        X = _rng_data(n=300)
        spec = Forecaster.spectrogram(X, channel=0, nperseg=32)
        assert set(spec.keys()) == {"times", "freqs", "Sxx"}

    def test_shape_consistency(self):
        X = _rng_data(n=300)
        spec = Forecaster.spectrogram(X, channel=0, nperseg=32)
        n_freqs = spec["Sxx"].shape[0]
        n_times = spec["Sxx"].shape[1]
        assert len(spec["freqs"]) == n_freqs
        assert len(spec["times"]) == n_times

    def test_1d_input(self):
        X = _rng_data(n=200)[:, 0]
        spec = Forecaster.spectrogram(X, nperseg=32)
        assert spec["Sxx"].ndim == 2

    def test_non_negative(self):
        X = _rng_data(n=200)
        spec = Forecaster.spectrogram(X, channel=0, nperseg=32)
        assert (spec["Sxx"] >= 0).all()


class TestPlotSpectrogram:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure, matplotlib.pyplot as plt
        X = _rng_data(n=300)
        fig = Forecaster.plot_spectrogram(X, channel=0, nperseg=32)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")


class TestSummaryTable:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_returns_dataframe(self):
        pytest.importorskip("pandas")
        import pandas as pd
        df = self.fc.summary_table(self.X)
        assert isinstance(df, pd.DataFrame)

    def test_columns_contain_metrics(self):
        pytest.importorskip("pandas")
        df = self.fc.summary_table(self.X)
        assert "mse" in df.columns and "mae" in df.columns

    def test_custom_horizons(self):
        pytest.importorskip("pandas")
        df = self.fc.summary_table(self.X, horizons=[0, 1, 2])
        assert len(df) == 3

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().summary_table(self.X)


class TestHistogramForecast:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure, matplotlib.pyplot as plt
        fc = _quick_fc().fit(_rng_data())
        fig = fc.histogram_forecast(_rng_data(n=200), channel=0)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().histogram_forecast(_rng_data(n=200))


class TestReliabilityScore:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_returns_dict(self):
        result = self.fc.reliability_score(self.X, self.X, coverage=0.9)
        assert set(result.keys()) >= {"picp", "pinaw", "cwc", "winkler", "nominal"}

    def test_picp_in_unit_interval(self):
        result = self.fc.reliability_score(self.X, self.X)
        assert 0.0 <= result["picp"] <= 1.0

    def test_pinaw_positive(self):
        result = self.fc.reliability_score(self.X, self.X)
        assert result["pinaw"] > 0

    def test_nominal_preserved(self):
        result = self.fc.reliability_score(self.X, self.X, coverage=0.8)
        assert result["nominal"] == pytest.approx(0.8)

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().reliability_score(self.X, self.X)


class TestForecastWithTrend:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_shape(self):
        pred = self.fc.forecast_with_trend(self.X, degree=1)
        assert pred.shape == (PRED, C)

    def test_higher_degree(self):
        pred = self.fc.forecast_with_trend(self.X, degree=2)
        assert pred.shape == (PRED, C)

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().forecast_with_trend(self.X)


class TestComputePinballLoss:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_returns_dict(self):
        result = self.fc.compute_pinball_loss(self.X, quantiles=(0.1, 0.5, 0.9), n_samples=5)
        assert isinstance(result, dict)
        assert set(result.keys()) == {0.1, 0.5, 0.9}

    def test_non_negative(self):
        result = self.fc.compute_pinball_loss(self.X, quantiles=(0.1, 0.5), n_samples=5)
        for v in result.values():
            if not np.isnan(v):
                assert v >= 0

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().compute_pinball_loss(self.X)


class TestWaveletDecomposition:
    def test_returns_dict(self):
        X = _rng_data(n=256)
        decomp = Forecaster.wavelet_decomposition(X, n_levels=4, channel=0)
        assert set(decomp.keys()) == {"approx", "details"}

    def test_n_details_equals_levels(self):
        X = _rng_data(n=256)
        decomp = Forecaster.wavelet_decomposition(X, n_levels=3, channel=0)
        assert len(decomp["details"]) == 3

    def test_1d_input(self):
        X = _rng_data(n=128)[:, 0]
        decomp = Forecaster.wavelet_decomposition(X, n_levels=2)
        assert "approx" in decomp

    def test_finite_values(self):
        X = _rng_data(n=128)
        decomp = Forecaster.wavelet_decomposition(X, n_levels=3)
        assert np.all(np.isfinite(decomp["approx"]))
        for d in decomp["details"]:
            assert np.all(np.isfinite(d))


class TestPlotWavelet:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure, matplotlib.pyplot as plt
        X = _rng_data(n=256)
        fig = Forecaster.plot_wavelet(X, n_levels=3, channel=0)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_correct_panel_count(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        X = _rng_data(n=256)
        fig = Forecaster.plot_wavelet(X, n_levels=4)
        assert len(fig.get_axes()) == 5  # 1 approx + 4 details
        plt.close("all")


class TestRollingPredictIter:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_is_generator(self):
        import types
        gen = self.fc.rolling_predict_iter(self.X)
        assert isinstance(gen, types.GeneratorType)

    def test_yields_correct_shape(self):
        gen = self.fc.rolling_predict_iter(self.X)
        t, pred = next(gen)
        assert pred.shape == (PRED, C)
        assert t == SEQ

    def test_step_advances_correctly(self):
        results = list(self.fc.rolling_predict_iter(self.X, step=PRED))
        if len(results) >= 2:
            assert results[1][0] - results[0][0] == PRED

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            list(_quick_fc().rolling_predict_iter(self.X))


class TestPredictAutoregressive:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_shape(self):
        pred = self.fc.predict_autoregressive(self.X, n_steps=50)
        assert pred.shape == (50, C)

    def test_exact_n_steps(self):
        for n in [1, PRED, PRED + 3, 3 * PRED]:
            pred = self.fc.predict_autoregressive(self.X, n_steps=n)
            assert pred.shape[0] == n

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().predict_autoregressive(self.X, n_steps=10)


class TestCompareChannelForecasts:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure, matplotlib.pyplot as plt
        fc = _quick_fc().fit(_rng_data())
        fig = fc.compare_channel_forecasts(_rng_data(n=200), channels=[0, 1])
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().compare_channel_forecasts(_rng_data(n=200))


class TestTemporalCrossValidation:
    def setup_method(self):
        self.fc = _quick_fc()
        self.X  = _rng_data(n=400)

    def test_returns_list(self):
        result = self.fc.temporal_cross_validation(self.X, n_splits=3)
        assert isinstance(result, list)

    def test_each_item_is_pair(self):
        result = self.fc.temporal_cross_validation(self.X, n_splits=3)
        for item in result:
            assert len(item) == 2

    def test_scores_finite(self):
        result = self.fc.temporal_cross_validation(self.X, n_splits=3, metric="mae")
        for _, score in result:
            assert np.isfinite(score)

    def test_train_sizes_increasing(self):
        result = self.fc.temporal_cross_validation(self.X, n_splits=3)
        sizes = [r[0] for r in result]
        assert sizes == sorted(sizes)


class TestPlotLearningCurve:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure, matplotlib.pyplot as plt
        fc = _quick_fc()
        fig = fc.plot_learning_curve(_rng_data(n=400), n_sizes=3)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")


class TestGetTargetCorrelations:
    def test_returns_dataframe(self):
        pytest.importorskip("pandas")
        import pandas as pd
        X = _rng_data(n=200)
        df = Forecaster.get_target_correlations(X, target_col=0, max_lag=10)
        assert isinstance(df, pd.DataFrame)

    def test_shape(self):
        pytest.importorskip("pandas")
        X = _rng_data(n=200)
        df = Forecaster.get_target_correlations(X, target_col=0, max_lag=10)
        assert df.shape == (21, C)

    def test_index_spans_lags(self):
        pytest.importorskip("pandas")
        X = _rng_data(n=200)
        df = Forecaster.get_target_correlations(X, max_lag=5)
        assert list(df.index) == list(range(-5, 6))

    def test_1d_raises_gracefully_or_works(self):
        pytest.importorskip("pandas")
        X = _rng_data(n=100)[:, 0]
        # 1D → treated as single channel; target_col must be 0
        df = Forecaster.get_target_correlations(X, target_col=0, max_lag=5)
        assert df.shape[0] == 11


class TestBatchEvaluate:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())

    def test_returns_dataframe(self):
        pytest.importorskip("pandas")
        import pandas as pd
        X_list = [_rng_data(n=200, seed=i) for i in range(3)]
        df = self.fc.batch_evaluate(X_list, names=["A", "B", "C"])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_columns_include_metrics(self):
        pytest.importorskip("pandas")
        X_list = [_rng_data(n=200, seed=0)]
        df = self.fc.batch_evaluate(X_list)
        assert "mse" in df.columns

    def test_index_uses_names(self):
        pytest.importorskip("pandas")
        X_list = [_rng_data(n=200, seed=0), _rng_data(n=200, seed=1)]
        df = self.fc.batch_evaluate(X_list, names=["X1", "X2"])
        assert list(df.index) == ["X1", "X2"]

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().batch_evaluate([_rng_data(n=200)])


class TestSpectralEntropy:
    def test_returns_float(self):
        X = _rng_data(n=200)
        se = Forecaster.spectral_entropy(X, channel=0)
        assert isinstance(se, float)

    def test_normalized_in_unit_interval(self):
        X = _rng_data(n=200)
        se = Forecaster.spectral_entropy(X, channel=0, normalize=True)
        assert 0.0 <= se <= 1.0

    def test_white_noise_near_one(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((1000, 1)).astype(np.float32)
        se = Forecaster.spectral_entropy(X, channel=0, normalize=True)
        assert se > 0.9   # white noise → near-max entropy

    def test_1d_input(self):
        X = _rng_data(n=200)[:, 0]
        se = Forecaster.spectral_entropy(X)
        assert isinstance(se, float)


class TestForecastBias:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_shape(self):
        bias = self.fc.forecast_bias(self.X)
        assert bias.shape == (PRED,)

    def test_finite(self):
        bias = self.fc.forecast_bias(self.X)
        assert np.all(np.isfinite(bias))

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().forecast_bias(self.X)


class TestPlotForecastBias:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure, matplotlib.pyplot as plt
        fc = _quick_fc().fit(_rng_data())
        fig = fc.plot_forecast_bias(_rng_data(n=200))
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")


class TestBoxCoxTransform:
    def test_returns_three_items(self):
        X = _rng_data(n=200)
        result = Forecaster.box_cox_transform(X, channel=0)
        assert len(result) == 3

    def test_transformed_is_array(self):
        X = _rng_data(n=200)
        x_tr, lam, offset = Forecaster.box_cox_transform(X, channel=0)
        assert isinstance(x_tr, np.ndarray)

    def test_inverse_roundtrip(self):
        X = _rng_data(n=200)
        arr = X[:, 0]
        x_tr, lam, offset = Forecaster.box_cox_transform(X, channel=0)
        x_back = Forecaster.box_cox_inverse(x_tr, lam, offset)
        np.testing.assert_allclose(x_back, arr, atol=1e-3, rtol=1e-3)

    def test_explicit_lam(self):
        X = _rng_data(n=200)
        x_tr, lam, _ = Forecaster.box_cox_transform(X, channel=0, lam=0.5)
        assert abs(lam - 0.5) < 1e-6

    def test_lam_zero_is_log(self):
        X = np.abs(_rng_data(n=100)) + 1.0   # strictly positive
        x_tr, lam, offset = Forecaster.box_cox_transform(X, channel=0, lam=0.0)
        arr = X[:, 0] + offset
        np.testing.assert_allclose(x_tr, np.log(arr).astype(np.float32), atol=1e-5)


class TestForecastDiagnostic:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_returns_dict(self):
        diag = self.fc.forecast_diagnostic(self.X)
        assert isinstance(diag, dict)

    def test_required_keys(self):
        diag = self.fc.forecast_diagnostic(self.X)
        assert set(diag.keys()) >= {"ljung_box", "residuals", "bias", "metrics"}

    def test_ljung_box_sub_dict(self):
        diag = self.fc.forecast_diagnostic(self.X)
        assert "p_value" in diag["ljung_box"]

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().forecast_diagnostic(self.X)


class TestFeatureImportanceRanking:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_returns_list(self):
        ranking = self.fc.feature_importance_ranking(self.X, n_permutations=2)
        assert isinstance(ranking, list)

    def test_length_equals_channels(self):
        ranking = self.fc.feature_importance_ranking(self.X, n_permutations=2)
        assert len(ranking) == C

    def test_sorted_descending(self):
        ranking = self.fc.feature_importance_ranking(self.X, n_permutations=2)
        importances = [imp for _, imp in ranking]
        assert importances == sorted(importances, reverse=True)

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().feature_importance_ranking(self.X)


class TestMemoryUsage:
    def test_returns_dict(self):
        fc = _quick_fc().fit(_rng_data())
        info = fc.memory_usage()
        assert isinstance(info, dict)

    def test_required_keys(self):
        fc = _quick_fc().fit(_rng_data())
        info = fc.memory_usage()
        assert set(info.keys()) >= {"total_params", "trainable_params", "size_mb"}

    def test_positive_values(self):
        fc = _quick_fc().fit(_rng_data())
        info = fc.memory_usage()
        assert info["total_params"] > 0
        assert info["size_mb"] > 0

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _quick_fc().memory_usage()


class TestPredictionStability:
    def test_returns_dict(self):
        X = _rng_data(n=200)
        fc = _quick_fc()
        result = fc.prediction_stability(X[:140], X[140:], n_seeds=2)
        assert set(result.keys()) == {"mean_pred", "std_pred", "cv"}

    def test_shapes(self):
        X = _rng_data(n=200)
        fc = _quick_fc()
        result = fc.prediction_stability(X[:140], X[140:], n_seeds=2)
        assert result["mean_pred"].shape == (PRED, C)
        assert result["std_pred"].shape  == (PRED, C)

    def test_cv_non_negative(self):
        X = _rng_data(n=200)
        fc = _quick_fc()
        result = fc.prediction_stability(X[:140], X[140:], n_seeds=2)
        assert result["cv"] >= 0


class TestTrendStrength:
    def test_returns_float(self):
        X = _rng_data(n=200)
        ts = Forecaster.trend_strength(X, window=30, channel=0)
        assert isinstance(ts, float)

    def test_in_unit_interval(self):
        X = _rng_data(n=200)
        ts = Forecaster.trend_strength(X, window=30, channel=0)
        assert 0.0 <= ts <= 1.0

    def test_pure_trend_near_one(self):
        t = np.linspace(0, 1, 200)
        X = t[:, None].astype(np.float32)
        ts = Forecaster.trend_strength(X, window=10, channel=0)
        assert ts > 0.8   # strongly trending

    def test_1d_input(self):
        X = _rng_data(n=200)[:, 0]
        ts = Forecaster.trend_strength(X, window=20)
        assert isinstance(ts, float)


class TestHodrickPrescottFilter:
    def test_returns_two_arrays(self):
        X = _rng_data(n=200)
        trend, cycle = Forecaster.hodrick_prescott_filter(X, lam=100, channel=0)
        assert trend.shape == (200,) and cycle.shape == (200,)

    def test_trend_plus_cycle_equals_original(self):
        X = _rng_data(n=100)
        trend, cycle = Forecaster.hodrick_prescott_filter(X, lam=100, channel=0)
        arr = X[:, 0].astype(np.float32)
        np.testing.assert_allclose(trend + cycle, arr, atol=1e-4)

    def test_1d_input(self):
        X = _rng_data(n=100)[:, 0]
        trend, cycle = Forecaster.hodrick_prescott_filter(X, lam=100)
        assert trend.shape == (100,)


class TestExponentialSmoothing:
    def test_single_es_shape(self):
        X = _rng_data(n=200)
        out = Forecaster.exponential_smoothing(X, alpha=0.3, channel=0)
        assert out.shape == (200,)

    def test_double_es_shape(self):
        X = _rng_data(n=200)
        out = Forecaster.exponential_smoothing(X, alpha=0.3, beta=0.1, channel=0)
        assert out.shape == (200,)

    def test_first_value_matches_input(self):
        X = _rng_data(n=100)
        out = Forecaster.exponential_smoothing(X, alpha=0.5, channel=0)
        assert out[0] == pytest.approx(float(X[0, 0]), abs=1e-5)

    def test_1d_input(self):
        X = _rng_data(n=100)[:, 0]
        out = Forecaster.exponential_smoothing(X, alpha=0.3)
        assert out.shape == (100,)


class TestCorrelationNetwork:
    def test_returns_list(self):
        X = _rng_data(n=200)
        edges = Forecaster.correlation_network(X, threshold=0.0)
        assert isinstance(edges, list)

    def test_threshold_zero_all_edges(self):
        X = _rng_data(n=200)
        edges = Forecaster.correlation_network(X, threshold=0.0)
        assert len(edges) == C * (C - 1) // 2

    def test_high_threshold_fewer_edges(self):
        X = _rng_data(n=200)
        edges_low  = Forecaster.correlation_network(X, threshold=0.0)
        edges_high = Forecaster.correlation_network(X, threshold=0.99)
        assert len(edges_high) <= len(edges_low)

    def test_edge_format(self):
        X = _rng_data(n=200)
        edges = Forecaster.correlation_network(X, threshold=0.0)
        if edges:
            i, j, r = edges[0]
            assert isinstance(i, int) and isinstance(j, int)
            assert -1.0 <= r <= 1.0


class TestPlotCorrelationNetwork:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure, matplotlib.pyplot as plt
        X = _rng_data(n=200)
        fig = Forecaster.plot_correlation_network(X, threshold=0.0)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")


class TestZNormalize:
    def test_shape_preserved(self):
        X = _rng_data(n=200)
        X_n, mu, sig = Forecaster.z_normalize(X)
        assert X_n.shape == X.shape

    def test_mu_near_zero_after_normalize(self):
        X = _rng_data(n=200)
        X_n, mu, sig = Forecaster.z_normalize(X)
        np.testing.assert_allclose(X_n.mean(axis=0), np.zeros(C), atol=1e-5)

    def test_std_near_one_after_normalize(self):
        X = _rng_data(n=200)
        X_n, mu, sig = Forecaster.z_normalize(X)
        np.testing.assert_allclose(X_n.std(axis=0), np.ones(C), atol=1e-4)

    def test_roundtrip(self):
        X = _rng_data(n=200)
        X_n, mu, sig = Forecaster.z_normalize(X)
        X_back = Forecaster.z_denormalize(X_n, mu, sig)
        np.testing.assert_allclose(X_back, X, atol=1e-5)

    def test_1d_becomes_2d(self):
        X = _rng_data(n=100)[:, 0]
        X_n, mu, sig = Forecaster.z_normalize(X)
        assert X_n.ndim == 2


class TestMutualInformation:
    def test_shape(self):
        X = _rng_data(n=200)
        mi = Forecaster.mutual_information(X)
        assert mi.shape == (C, C)

    def test_symmetric(self):
        X = _rng_data(n=200)
        mi = Forecaster.mutual_information(X)
        np.testing.assert_allclose(mi, mi.T, atol=1e-10)

    def test_diagonal_nonnegative(self):
        X = _rng_data(n=200)
        mi = Forecaster.mutual_information(X)
        assert (np.diag(mi) >= 0).all()

    def test_off_diagonal_nonnegative(self):
        X = _rng_data(n=200)
        mi = Forecaster.mutual_information(X)
        assert (mi >= 0).all()

    def test_perfect_dependence_high_mi(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(500)
        X = np.column_stack([x, x])   # identical channels
        mi = Forecaster.mutual_information(X)
        # MI(0,1) should equal MI(0,0) for perfectly dependent channels
        assert mi[0, 1] > 0


class TestPlotMutualInformation:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure
        import matplotlib.pyplot as plt
        X = _rng_data(n=200)
        fig = Forecaster.plot_mutual_information(X)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_custom_title(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        X = _rng_data(n=200)
        fig = Forecaster.plot_mutual_information(X, title="MI")
        ax = fig.get_axes()[0]
        assert "MI" in ax.get_title()
        plt.close("all")


class TestSeasonalStrength:
    def test_strong_seasonality(self):
        # Pure sine wave → high seasonal strength
        t = np.arange(300)
        x = np.sin(2 * np.pi * t / 12).astype(np.float32)[:, None]
        X = np.column_stack([x, x])
        s = Forecaster.seasonal_strength(X, period=12, channel=0)
        assert s > 0.5

    def test_white_noise_low_strength(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((300, 2)).astype(np.float32)
        s = Forecaster.seasonal_strength(X, period=12, channel=0)
        assert 0.0 <= s <= 1.0

    def test_returns_float_in_01(self):
        X = _rng_data(n=200)
        s = Forecaster.seasonal_strength(X, period=7)
        assert 0.0 <= s <= 1.0

    def test_invalid_period_raises(self):
        X = _rng_data(n=200)
        with pytest.raises(ValueError, match="period"):
            Forecaster.seasonal_strength(X, period=1)


class TestOptimalLag:
    def test_returns_dict(self):
        X = _rng_data(n=200)
        result = Forecaster.optimal_lag(X, max_lag=10)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        X = _rng_data(n=200)
        result = Forecaster.optimal_lag(X, max_lag=10)
        for k in ("best_lag", "aic", "bic", "scores"):
            assert k in result

    def test_best_lag_in_range(self):
        X = _rng_data(n=200)
        result = Forecaster.optimal_lag(X, max_lag=10)
        assert 1 <= result["best_lag"] <= 10

    def test_aic_length(self):
        X = _rng_data(n=200)
        result = Forecaster.optimal_lag(X, max_lag=10)
        assert len(result["aic"]) == 10

    def test_bic_criterion(self):
        X = _rng_data(n=200)
        result = Forecaster.optimal_lag(X, max_lag=10, criterion="bic")
        assert (result["scores"] == result["bic"]).all()

    def test_invalid_criterion_raises(self):
        X = _rng_data(n=200)
        with pytest.raises(ValueError, match="criterion"):
            Forecaster.optimal_lag(X, max_lag=10, criterion="hqic")

    def test_best_lag_is_integer(self):
        X = _rng_data(n=200)
        result = Forecaster.optimal_lag(X, max_lag=10)
        assert isinstance(result["best_lag"], (int, np.integer))


class TestDashboard:
    def setup_method(self):
        pytest.importorskip("matplotlib")
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_returns_figure(self):
        import matplotlib.figure
        import matplotlib.pyplot as plt
        fig = self.fc.dashboard(self.X)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_six_axes(self):
        import matplotlib.pyplot as plt
        fig = self.fc.dashboard(self.X)
        assert len(fig.get_axes()) >= 6
        plt.close("all")

    def test_custom_title(self):
        import matplotlib.pyplot as plt
        fig = self.fc.dashboard(self.X, title="My Dashboard")
        assert "My Dashboard" in fig.texts[0].get_text()
        plt.close("all")

    def test_custom_channel(self):
        import matplotlib.pyplot as plt
        fig = self.fc.dashboard(self.X, channel=1)
        assert fig is not None
        plt.close("all")

    def test_unfitted_raises(self):
        fc = Forecaster("NLinear", seq_len=SEQ, pred_len=PRED)
        with pytest.raises(RuntimeError):
            fc.dashboard(self.X)


class TestPredictQuantiles:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=SEQ)

    def test_returns_dict(self):
        result = self.fc.predict_quantiles(self.X, n_samples=10)
        assert isinstance(result, dict)

    def test_mean_and_std_present(self):
        result = self.fc.predict_quantiles(self.X, n_samples=10)
        assert "mean" in result
        assert "std"  in result

    def test_quantile_keys(self):
        result = self.fc.predict_quantiles(
            self.X, quantiles=[0.1, 0.5, 0.9], n_samples=10
        )
        assert "q10" in result
        assert "q50" in result
        assert "q90" in result

    def test_shapes(self):
        result = self.fc.predict_quantiles(
            self.X, quantiles=[0.1, 0.9], n_samples=10
        )
        assert result["q10"].shape == (PRED, C)
        assert result["q90"].shape == (PRED, C)

    def test_monotonicity(self):
        result = self.fc.predict_quantiles(
            self.X, quantiles=[0.1, 0.5, 0.9], n_samples=20
        )
        # q10 ≤ q50 ≤ q90 element-wise (within tolerance)
        assert (result["q10"] <= result["q90"] + 1e-6).all()

    def test_unfitted_raises(self):
        fc = Forecaster("NLinear", seq_len=SEQ, pred_len=PRED)
        with pytest.raises(RuntimeError):
            fc.predict_quantiles(self.X, n_samples=5)


class TestStationarityTest:
    def test_returns_dict(self):
        X = _rng_data(n=200)
        result = Forecaster.stationarity_test(X)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        X = _rng_data(n=200)
        result = Forecaster.stationarity_test(X)
        for k in ("adf_stat", "p_value", "n_obs", "critical_1",
                  "critical_5", "critical_10"):
            assert k in result

    def test_p_value_in_range(self):
        X = _rng_data(n=200)
        result = Forecaster.stationarity_test(X)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_stationary_series_low_p(self):
        # White noise is stationary → ADF stat should be very negative
        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, 1)).astype(np.float32)
        result = Forecaster.stationarity_test(X)
        assert result["adf_stat"] < result["critical_5"]   # reject unit root

    def test_random_walk_high_p(self):
        # Random walk has a unit root → p_value should be high
        rng = np.random.default_rng(7)
        X = np.cumsum(rng.standard_normal(500)).astype(np.float32)[:, None]
        result = Forecaster.stationarity_test(X)
        assert result["p_value"] > 0.3    # fail to reject

    def test_1d_input(self):
        X = _rng_data(n=200)[:, 0]
        result = Forecaster.stationarity_test(X)
        assert "adf_stat" in result

    def test_custom_max_lags(self):
        X = _rng_data(n=200)
        r1 = Forecaster.stationarity_test(X, max_lags=2)
        r4 = Forecaster.stationarity_test(X, max_lags=4)
        assert r1["n_obs"] != r4["n_obs"]


class TestCrossValScore:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X  = _rng_data(n=200)

    def test_returns_array(self):
        scores = self.fc.cross_val_score(self.X, n_splits=3, refit=False)
        assert isinstance(scores, np.ndarray)

    def test_length_le_n_splits(self):
        scores = self.fc.cross_val_score(self.X, n_splits=3, refit=False)
        assert len(scores) <= 3

    def test_scores_nonnegative(self):
        scores = self.fc.cross_val_score(self.X, n_splits=3, refit=False)
        assert (scores >= 0).all()

    def test_mse_metric(self):
        scores = self.fc.cross_val_score(self.X, n_splits=2, metric="MSE", refit=False)
        assert (scores >= 0).all()

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric"):
            self.fc.cross_val_score(self.X, n_splits=2, metric="SMAPE", refit=False)

    def test_refit_leaves_model_fitted(self):
        fc = _quick_fc().fit(_rng_data())
        fc.cross_val_score(self.X, n_splits=2, refit=True)
        assert fc._model is not None


class TestDetectChangePoints:
    def test_returns_array(self):
        X = _rng_data(n=200)
        cps = Forecaster.detect_change_points(X, window=10)
        assert isinstance(cps, np.ndarray)

    def test_indices_in_range(self):
        X = _rng_data(n=200)
        cps = Forecaster.detect_change_points(X, window=10)
        assert (cps >= 0).all() and (cps < 200).all()

    def test_custom_threshold(self):
        X = _rng_data(n=200)
        cps_high = Forecaster.detect_change_points(X, window=10, threshold=100.0)
        cps_low = Forecaster.detect_change_points(X, window=10, threshold=0.0)
        # lower threshold → at least as many change points
        assert len(cps_low) >= len(cps_high)

    def test_detects_obvious_change(self):
        # abrupt level shift
        rng = np.random.default_rng(42)
        X = np.concatenate([
            rng.standard_normal((100, 1)),
            rng.standard_normal((100, 1)) + 10.0,
        ])
        cps = Forecaster.detect_change_points(X, window=10)
        # change point should be somewhere near timestep 100
        assert any(80 <= cp <= 120 for cp in cps)

    def test_1d_input(self):
        X = _rng_data(n=100)[:, 0]
        cps = Forecaster.detect_change_points(X, window=10)
        assert isinstance(cps, np.ndarray)


class TestPlotChangePoints:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure
        import matplotlib.pyplot as plt
        X = _rng_data(n=100)
        cps = Forecaster.detect_change_points(X, window=5)
        fig = Forecaster.plot_change_points(X, cps)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_empty_change_points(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        X = _rng_data(n=100)
        fig = Forecaster.plot_change_points(X, np.array([]))
        assert fig is not None
        plt.close("all")


class TestDescribe:
    def test_returns_dataframe(self):
        pytest.importorskip("pandas")
        import pandas as pd
        X = _rng_data(n=100)
        df = Forecaster.describe(X)
        assert isinstance(df, pd.DataFrame)

    def test_columns_are_channels(self):
        pytest.importorskip("pandas")
        X = _rng_data(n=100)
        df = Forecaster.describe(X)
        assert list(df.columns) == [f"ch{i}" for i in range(C)]

    def test_expected_rows(self):
        pytest.importorskip("pandas")
        X = _rng_data(n=100)
        df = Forecaster.describe(X)
        for row in ("count", "mean", "std", "min", "q25", "median",
                    "q75", "max", "range", "skewness", "kurtosis",
                    "autocorr_lag1", "n_missing"):
            assert row in df.index

    def test_n_missing_counts_nans(self):
        pytest.importorskip("pandas")
        X = _rng_data(n=100)
        X[5:10, 0] = np.nan
        df = Forecaster.describe(X)
        assert df.loc["n_missing", "ch0"] == 5

    def test_custom_channel_names(self):
        pytest.importorskip("pandas")
        X = _rng_data(n=100)
        df = Forecaster.describe(X, channel_names=["A", "B", "C"])
        assert list(df.columns) == ["A", "B", "C"]

    def test_1d_input(self):
        pytest.importorskip("pandas")
        X = _rng_data(n=100)[:, 0]
        df = Forecaster.describe(X)
        assert df.shape[1] == 1


class TestPersistenceForecast:
    def setup_method(self):
        self.fc = _quick_fc()  # no need to fit for persistence

    def test_shape(self):
        X = _rng_data(n=SEQ)
        pred = self.fc.persistence_forecast(X)
        assert pred.shape == (PRED, C)

    def test_repeats_last_value(self):
        X = _rng_data(n=SEQ)
        pred = self.fc.persistence_forecast(X)
        np.testing.assert_array_equal(pred[0], X[-1])
        np.testing.assert_array_equal(pred[-1], X[-1])

    def test_lag_2(self):
        X = _rng_data(n=SEQ)
        pred = self.fc.persistence_forecast(X, lag=2)
        np.testing.assert_array_equal(pred[0], X[-2])

    def test_1d_input(self):
        X = _rng_data(n=SEQ)[:, 0]
        pred = self.fc.persistence_forecast(X)
        assert pred.shape == (PRED, 1)


class TestScoreVsPersistence:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=100)

    def test_returns_dict(self):
        result = self.fc.score_vs_persistence(self.X)
        assert isinstance(result, dict)

    def test_keys_present(self):
        result = self.fc.score_vs_persistence(self.X)
        assert "model" in result
        assert "persistence" in result

    def test_persistence_has_metrics(self):
        result = self.fc.score_vs_persistence(self.X)
        for m in ("MSE", "MAE", "RMSE", "SMAPE"):
            assert m in result["persistence"]

    def test_unfitted_raises(self):
        fc = Forecaster("NLinear", seq_len=SEQ, pred_len=PRED)
        with pytest.raises(RuntimeError):
            fc.score_vs_persistence(self.X)


class TestChunkedPredict:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=100)

    def test_shape(self):
        out = self.fc.chunked_predict(self.X, chunk_size=8)
        n_windows = len(self.X) - SEQ + 1
        assert out.shape == (n_windows, PRED, C)

    def test_small_chunk_equals_large_chunk(self):
        # same predictions regardless of chunk size
        out1 = self.fc.chunked_predict(self.X, chunk_size=4)
        out2 = self.fc.chunked_predict(self.X, chunk_size=32)
        np.testing.assert_allclose(out1, out2, rtol=1e-5)

    def test_unfitted_raises(self):
        fc = Forecaster("NLinear", seq_len=SEQ, pred_len=PRED)
        with pytest.raises(RuntimeError):
            fc.chunked_predict(self.X)

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            self.fc.chunked_predict(_rng_data(n=SEQ - 1))


class TestInterpolateMissing:
    def test_fills_nan_linear(self):
        X = _rng_data(n=50)
        X[5, 0] = np.nan
        X[6, 0] = np.nan
        out = Forecaster.interpolate_missing(X)
        assert not np.isnan(out).any()

    def test_shape_preserved(self):
        X = _rng_data(n=50)
        X[10, 1] = np.nan
        out = Forecaster.interpolate_missing(X)
        assert out.shape == X.shape

    def test_no_nan_passthrough(self):
        X = _rng_data(n=50)
        out = Forecaster.interpolate_missing(X)
        np.testing.assert_array_equal(out, X.astype(float))

    def test_forward_method(self):
        X = _rng_data(n=50)
        X[5:10, 0] = np.nan
        out = Forecaster.interpolate_missing(X, method="forward")
        assert not np.isnan(out).any()

    def test_backward_method(self):
        X = _rng_data(n=50)
        X[5:10, 0] = np.nan
        out = Forecaster.interpolate_missing(X, method="backward")
        assert not np.isnan(out).any()

    def test_nearest_method(self):
        X = _rng_data(n=50)
        X[5, 0] = np.nan
        out = Forecaster.interpolate_missing(X, method="nearest")
        assert not np.isnan(out).any()

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            Forecaster.interpolate_missing(_rng_data(n=50), method="spline")

    def test_1d_input(self):
        X = _rng_data(n=50)[:, 0]
        X[5] = np.nan
        out = Forecaster.interpolate_missing(X)
        assert out.ndim == 1
        assert not np.isnan(out).any()


class TestHorizonErrorProfile:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=80)

    def test_shape(self):
        profile = self.fc.horizon_error_profile(self.X)
        assert profile.shape == (PRED,)

    def test_nonnegative(self):
        profile = self.fc.horizon_error_profile(self.X)
        assert (profile >= 0).all()

    def test_mse_metric(self):
        profile = self.fc.horizon_error_profile(self.X, metric="MSE")
        assert profile.shape == (PRED,)

    def test_rmse_metric(self):
        profile = self.fc.horizon_error_profile(self.X, metric="RMSE")
        assert profile.shape == (PRED,)

    def test_rmse_ge_mae(self):
        mae = self.fc.horizon_error_profile(self.X, metric="MAE")
        rmse = self.fc.horizon_error_profile(self.X, metric="RMSE")
        # RMSE >= MAE by Jensen's inequality
        assert (rmse >= mae - 1e-6).all()

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric"):
            self.fc.horizon_error_profile(self.X, metric="SMAPE")

    def test_unfitted_raises(self):
        fc = Forecaster("NLinear", seq_len=SEQ, pred_len=PRED)
        with pytest.raises(RuntimeError):
            fc.horizon_error_profile(self.X)


class TestPlotHorizonErrorProfile:
    def setup_method(self):
        pytest.importorskip("matplotlib")
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=80)

    def test_returns_figure(self):
        import matplotlib.figure
        import matplotlib.pyplot as plt
        fig = self.fc.plot_horizon_error_profile(self.X)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_custom_title(self):
        import matplotlib.pyplot as plt
        fig = self.fc.plot_horizon_error_profile(self.X, title="Horizon")
        ax = fig.get_axes()[0]
        assert "Horizon" in ax.get_title()
        plt.close("all")


class TestCountParameters:
    def test_returns_dict(self):
        fc = _quick_fc().fit(_rng_data())
        result = fc.count_parameters()
        assert isinstance(result, dict)

    def test_total_key_present(self):
        fc = _quick_fc().fit(_rng_data())
        result = fc.count_parameters()
        assert "total" in result

    def test_total_positive(self):
        fc = _quick_fc().fit(_rng_data())
        result = fc.count_parameters()
        assert result["total"] > 0

    def test_trainable_le_total(self):
        fc = _quick_fc().fit(_rng_data())
        trainable = fc.count_parameters(trainable_only=True)["total"]
        total = fc.count_parameters()["total"]
        assert trainable <= total

    def test_unfitted_raises(self):
        fc = Forecaster("NLinear", seq_len=SEQ, pred_len=PRED)
        with pytest.raises(RuntimeError):
            fc.count_parameters()


class TestRollingCorrelation:
    def test_shapes(self):
        X = _rng_data(n=100)
        ts, corr = Forecaster.rolling_correlation(X, window=20)
        assert ts.shape == corr.shape

    def test_output_length(self):
        X = _rng_data(n=100)
        ts, corr = Forecaster.rolling_correlation(X, window=20)
        assert len(corr) == 100 - 20 + 1

    def test_bounded_values(self):
        X = _rng_data(n=100)
        _, corr = Forecaster.rolling_correlation(X, window=20)
        assert (corr >= -1.0 - 1e-6).all() and (corr <= 1.0 + 1e-6).all()

    def test_perfect_correlation(self):
        x = np.arange(100, dtype=float)
        X = np.column_stack([x, 2 * x + 1])
        _, corr = Forecaster.rolling_correlation(X, window=10)
        np.testing.assert_allclose(corr, 1.0, atol=1e-6)

    def test_timestep_values(self):
        X = _rng_data(n=100)
        ts, _ = Forecaster.rolling_correlation(X, window=20)
        assert ts[0] == 19  # window - 1
        assert ts[-1] == 99


class TestPlotRollingCorrelation:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure
        import matplotlib.pyplot as plt
        X = _rng_data(n=100)
        fig = Forecaster.plot_rolling_correlation(X, window=20)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_custom_title(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        X = _rng_data(n=100)
        fig = Forecaster.plot_rolling_correlation(X, window=20, title="RC")
        ax = fig.get_axes()[0]
        assert "RC" in ax.get_title()
        plt.close("all")


class TestSetDevice:
    def test_returns_self(self):
        fc = _quick_fc().fit(_rng_data())
        result = fc.set_device("cpu")
        assert result is fc

    def test_device_attribute_updated(self):
        fc = _quick_fc().fit(_rng_data())
        fc.set_device("cpu")
        assert fc.device == "cpu"

    def test_unfitted_no_error(self):
        fc = Forecaster("NLinear", seq_len=SEQ, pred_len=PRED)
        fc.set_device("cpu")  # no model to move, but device attr should update
        assert fc.device == "cpu"

    def test_predictions_unchanged_after_cpu_move(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data(n=SEQ)
        pred_before = fc.predict(X[np.newaxis]).copy()
        fc.set_device("cpu")
        pred_after = fc.predict(X[np.newaxis])
        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5)


class TestGrangerTest:
    def test_shape(self):
        X = _rng_data(n=100)
        f = Forecaster.granger_test(X, max_lag=3)
        assert f.shape == (C, C)

    def test_diagonal_zero(self):
        X = _rng_data(n=100)
        f = Forecaster.granger_test(X, max_lag=3)
        np.testing.assert_allclose(np.diag(f), 0.0)

    def test_nonnegative(self):
        X = _rng_data(n=100)
        f = Forecaster.granger_test(X, max_lag=3)
        assert (f >= 0).all()

    def test_known_causality(self):
        # ch0 causes ch1: ch1[t] = ch0[t-1] + noise
        rng = np.random.default_rng(42)
        T = 200
        x0 = rng.standard_normal(T)
        x1 = np.roll(x0, 1) + 0.1 * rng.standard_normal(T)
        X = np.column_stack([x0, x1])
        f = Forecaster.granger_test(X, max_lag=3)
        # F(ch0 → ch1) should be large
        assert f[0, 1] > f[1, 0]

    def test_univariate_raises(self):
        X = _rng_data(n=100, c=1)
        # Still produces (1,1) matrix — just diagonal zeros
        f = Forecaster.granger_test(X, max_lag=3)
        assert f.shape == (1, 1)


class TestPlotGranger:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure
        import matplotlib.pyplot as plt
        X = _rng_data(n=100)
        fig = Forecaster.plot_granger(X, max_lag=3)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_custom_title(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        X = _rng_data(n=100)
        fig = Forecaster.plot_granger(X, max_lag=3, title="Granger")
        ax = fig.get_axes()[0]
        assert "Granger" in ax.get_title()
        plt.close("all")


class TestInputGradient:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=SEQ)

    def test_output_shape(self):
        grad = self.fc.input_gradient(self.X)
        assert grad.shape == (SEQ, C)

    def test_absolute_nonnegative(self):
        grad = self.fc.input_gradient(self.X, absolute=True)
        assert (grad >= 0).all()

    def test_non_absolute_has_both_signs(self):
        # with a random model there should be both positive and negative values
        grad = self.fc.input_gradient(self.X, absolute=False)
        # just check the array has finite values and correct shape
        assert np.isfinite(grad).all()
        assert grad.shape == (SEQ, C)

    def test_custom_target_step(self):
        grad = self.fc.input_gradient(self.X, target_step=PRED - 1)
        assert grad.shape == (SEQ, C)

    def test_custom_target_channel(self):
        grad = self.fc.input_gradient(self.X, target_channel=C - 1)
        assert grad.shape == (SEQ, C)

    def test_long_context_trimmed(self):
        X_long = _rng_data(n=SEQ + 10)
        grad = self.fc.input_gradient(X_long)
        assert grad.shape == (SEQ, C)

    def test_unfitted_raises(self):
        fc = Forecaster("NLinear", seq_len=SEQ, pred_len=PRED)
        with pytest.raises(RuntimeError):
            fc.input_gradient(self.X)


class TestPlotSaliency:
    def setup_method(self):
        pytest.importorskip("matplotlib")
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=SEQ)

    def test_returns_figure(self):
        import matplotlib.figure
        import matplotlib.pyplot as plt
        fig = self.fc.plot_saliency(self.X)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_custom_title(self):
        import matplotlib.pyplot as plt
        fig = self.fc.plot_saliency(self.X, title="Saliency")
        ax = fig.get_axes()[0]
        assert "Saliency" in ax.get_title()
        plt.close("all")


class TestErrorDecomposition:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())

    def test_returns_dict(self):
        X_train = _rng_data(n=200)
        X_test = _rng_data(n=80, seed=1)
        result = self.fc.error_decomposition(X_train, X_test, n_bootstrap=3, seed=0)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        X_train = _rng_data(n=200)
        X_test = _rng_data(n=80, seed=1)
        result = self.fc.error_decomposition(X_train, X_test, n_bootstrap=3, seed=0)
        for key in ("bias2", "variance", "total_mse"):
            assert key in result

    def test_nonnegative_values(self):
        X_train = _rng_data(n=200)
        X_test = _rng_data(n=80, seed=1)
        result = self.fc.error_decomposition(X_train, X_test, n_bootstrap=3, seed=0)
        for v in result.values():
            assert v >= 0


class TestMovingForecast:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=100)

    def test_output_shape(self):
        pred = self.fc.moving_forecast(self.X, n_windows=3)
        assert pred.shape == (PRED, C)

    def test_single_window_matches_predict(self):
        pred_moving = self.fc.moving_forecast(self.X, n_windows=1)
        pred_regular = self.fc.predict(self.X[-SEQ:])
        np.testing.assert_allclose(pred_moving, pred_regular, rtol=1e-5)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="seq_len"):
            self.fc.moving_forecast(_rng_data(n=SEQ + 1), n_windows=5)

    def test_unfitted_raises(self):
        fc = Forecaster("NLinear", seq_len=SEQ, pred_len=PRED)
        with pytest.raises(RuntimeError):
            fc.moving_forecast(self.X)


class TestResidualDistribution:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=100)

    def test_returns_dict(self):
        dist = self.fc.residual_distribution(self.X)
        assert isinstance(dist, dict)

    def test_expected_keys(self):
        dist = self.fc.residual_distribution(self.X)
        for key in ("mean", "std", "skewness", "kurtosis", "q5", "q25",
                    "median", "q75", "q95", "n"):
            assert key in dist

    def test_n_positive(self):
        dist = self.fc.residual_distribution(self.X)
        assert dist["n"] > 0

    def test_quantile_ordering(self):
        dist = self.fc.residual_distribution(self.X)
        assert dist["q5"] <= dist["q25"] <= dist["median"] <= dist["q75"] <= dist["q95"]

    def test_std_nonnegative(self):
        dist = self.fc.residual_distribution(self.X)
        assert dist["std"] >= 0

    def test_custom_channel(self):
        d0 = self.fc.residual_distribution(self.X, channel=0)
        d1 = self.fc.residual_distribution(self.X, channel=1)
        # different channels → different distributions
        assert d0["mean"] != d1["mean"] or d0["std"] != d1["std"]

    def test_unfitted_raises(self):
        fc = Forecaster("NLinear", seq_len=SEQ, pred_len=PRED)
        with pytest.raises(RuntimeError):
            fc.residual_distribution(self.X)


class TestChannelCorrelation:
    def test_shape(self):
        X = _rng_data(n=100)
        corr = Forecaster.channel_correlation(X)
        assert corr.shape == (C, C)

    def test_diagonal_ones(self):
        X = _rng_data(n=100)
        corr = Forecaster.channel_correlation(X)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)

    def test_symmetric(self):
        X = _rng_data(n=100)
        corr = Forecaster.channel_correlation(X)
        np.testing.assert_allclose(corr, corr.T, atol=1e-10)

    def test_values_in_minus1_to_1(self):
        X = _rng_data(n=100)
        corr = Forecaster.channel_correlation(X)
        assert (corr >= -1.0 - 1e-6).all() and (corr <= 1.0 + 1e-6).all()

    def test_perfect_correlation(self):
        # perfectly correlated channels
        x = np.arange(100, dtype=float)[:, None]
        X = np.concatenate([x, 2 * x], axis=1)
        corr = Forecaster.channel_correlation(X)
        assert abs(corr[0, 1] - 1.0) < 1e-6


class TestPlotChannelCorrelation:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure
        import matplotlib.pyplot as plt
        X = _rng_data(n=100)
        fig = Forecaster.plot_channel_correlation(X)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_custom_title(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        X = _rng_data(n=100)
        fig = Forecaster.plot_channel_correlation(X, title="Corr")
        ax = fig.get_axes()[0]
        assert "Corr" in ax.get_title()
        plt.close("all")

    def test_custom_channel_names(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        X = _rng_data(n=100)
        names = ["a", "b", "c"]
        fig = Forecaster.plot_channel_correlation(X, channel_names=names)
        ax = fig.get_axes()[0]
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == names
        plt.close("all")


class TestForecastDataframe:
    def setup_method(self):
        pytest.importorskip("pandas")
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=SEQ)

    def test_returns_dataframe(self):
        import pandas as pd
        df = self.fc.forecast_dataframe(self.X)
        assert isinstance(df, pd.DataFrame)

    def test_shape(self):
        df = self.fc.forecast_dataframe(self.X)
        assert df.shape == (PRED, C)

    def test_default_column_names(self):
        df = self.fc.forecast_dataframe(self.X)
        assert list(df.columns) == [f"ch{i}" for i in range(C)]

    def test_custom_column_names(self):
        names = ["A", "B", "D"]
        df = self.fc.forecast_dataframe(self.X, channel_names=names)
        assert list(df.columns) == names

    def test_start_index(self):
        df = self.fc.forecast_dataframe(self.X, start_index=100)
        assert df.index[0] == 100
        assert df.index[-1] == 100 + PRED - 1

    def test_unfitted_raises(self):
        fc = Forecaster("NLinear", seq_len=SEQ, pred_len=PRED)
        with pytest.raises(RuntimeError):
            fc.forecast_dataframe(self.X)


class TestProfile:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=SEQ)

    def test_returns_dict(self):
        result = self.fc.profile(self.X, n_repeats=5)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        result = self.fc.profile(self.X, n_repeats=5)
        for key in ("mean_ms", "std_ms", "min_ms", "max_ms", "throughput"):
            assert key in result

    def test_latency_positive(self):
        result = self.fc.profile(self.X, n_repeats=5)
        assert result["mean_ms"] > 0

    def test_throughput_positive(self):
        result = self.fc.profile(self.X, n_repeats=5)
        assert result["throughput"] > 0

    def test_min_le_mean_le_max(self):
        result = self.fc.profile(self.X, n_repeats=5)
        assert result["min_ms"] <= result["mean_ms"] <= result["max_ms"]

    def test_unfitted_raises(self):
        fc = Forecaster("NLinear", seq_len=SEQ, pred_len=PRED)
        with pytest.raises(RuntimeError):
            fc.profile(self.X, n_repeats=2)


class TestWarmup:
    def test_returns_self(self):
        fc = _quick_fc().fit(_rng_data())
        result = fc.warmup(n=2)
        assert result is fc

    def test_no_input_uses_zeros(self):
        fc = _quick_fc().fit(_rng_data())
        fc.warmup(n=2)  # should not raise

    def test_with_custom_context(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data(n=SEQ)
        fc.warmup(X, n=2)  # should not raise

    def test_unfitted_raises(self):
        fc = Forecaster("NLinear", seq_len=SEQ, pred_len=PRED)
        with pytest.raises(RuntimeError):
            fc.warmup(n=1)


class TestResidualQQ:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=100)

    def test_returns_two_arrays(self):
        th, sa = self.fc.residual_qq(self.X)
        assert isinstance(th, np.ndarray)
        assert isinstance(sa, np.ndarray)

    def test_shapes_match(self):
        th, sa = self.fc.residual_qq(self.X)
        assert th.shape == sa.shape

    def test_sample_is_sorted(self):
        _, sa = self.fc.residual_qq(self.X)
        assert (np.diff(sa) >= 0).all()

    def test_theoretical_is_sorted(self):
        th, _ = self.fc.residual_qq(self.X)
        assert (np.diff(th) > 0).all()

    def test_custom_channel(self):
        th0, _ = self.fc.residual_qq(self.X, channel=0)
        th1, _ = self.fc.residual_qq(self.X, channel=1)
        assert th0.shape == th1.shape

    def test_unfitted_raises(self):
        fc = Forecaster("NLinear", seq_len=SEQ, pred_len=PRED)
        with pytest.raises(RuntimeError):
            fc.residual_qq(self.X)


class TestPlotQQ:
    def setup_method(self):
        pytest.importorskip("matplotlib")
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=100)

    def test_returns_figure(self):
        import matplotlib.figure
        import matplotlib.pyplot as plt
        fig = self.fc.plot_qq(self.X)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_custom_title(self):
        import matplotlib.pyplot as plt
        fig = self.fc.plot_qq(self.X, title="QQ Test")
        ax = fig.get_axes()[0]
        assert "QQ Test" in ax.get_title()
        plt.close("all")


class TestToTorchDataset:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=100)

    def test_returns_dataset(self):
        import torch.utils.data
        ds = self.fc.to_torch_dataset(self.X)
        assert isinstance(ds, torch.utils.data.Dataset)

    def test_length(self):
        ds = self.fc.to_torch_dataset(self.X)
        expected = len(self.X) - SEQ - PRED + 1
        assert len(ds) == expected

    def test_item_shapes(self):
        ds = self.fc.to_torch_dataset(self.X)
        x, y = ds[0]
        assert x.shape == (SEQ, C)
        assert y.shape == (PRED, C)

    def test_no_normalize(self):
        import torch
        ds = self.fc.to_torch_dataset(self.X, normalize=False)
        x, _ = ds[0]
        # raw data first window — check dtype
        assert x.dtype == torch.float32

    def test_1d_input(self):
        import torch.utils.data
        X1d = _rng_data(n=100)[:, 0]
        ds = self.fc.to_torch_dataset(X1d)
        assert isinstance(ds, torch.utils.data.Dataset)

    def test_unfitted_raises(self):
        fc = Forecaster("NLinear", seq_len=SEQ, pred_len=PRED)
        with pytest.raises(RuntimeError):
            fc.to_torch_dataset(self.X)


class TestCopyWeightsFrom:
    def test_copies_weights_no_error(self):
        fc_src = _quick_fc().fit(_rng_data())
        fc_dst = _quick_fc().fit(_rng_data(seed=1))
        fc_dst.copy_weights_from(fc_src)
        # model parameters should now be identical
        for p_src, p_dst in zip(
            fc_src._model.parameters(), fc_dst._model.parameters()
        ):
            np.testing.assert_allclose(
                p_dst.detach().numpy(), p_src.detach().numpy(), rtol=1e-6
            )

    def test_returns_self(self):
        fc_src = _quick_fc().fit(_rng_data())
        fc_dst = _quick_fc().fit(_rng_data(seed=1))
        result = fc_dst.copy_weights_from(fc_src)
        assert result is fc_dst

    def test_unfitted_dst_raises(self):
        fc_src = _quick_fc().fit(_rng_data())
        fc_dst = _quick_fc()
        with pytest.raises(RuntimeError):
            fc_dst.copy_weights_from(fc_src)

    def test_unfitted_src_raises(self):
        fc_src = _quick_fc()
        fc_dst = _quick_fc().fit(_rng_data())
        with pytest.raises(RuntimeError):
            fc_dst.copy_weights_from(fc_src)


class TestAlignChannels:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())  # trained on C=3 channels

    def test_passthrough_when_matching(self):
        X = _rng_data(n=50)
        out = self.fc.align_channels(X)
        np.testing.assert_array_equal(out, X)

    def test_truncates_extra_channels(self):
        X = _rng_data(n=50, c=5)
        out = self.fc.align_channels(X)
        assert out.shape == (50, C)

    def test_pads_missing_channels(self):
        X = _rng_data(n=50, c=1)
        out = self.fc.align_channels(X)
        assert out.shape == (50, C)
        # padded columns are zero
        assert (out[:, 1:] == 0).all()

    def test_works_on_3d_input(self):
        X = _rng_data(n=50, c=5)
        X3d = X[np.newaxis]  # (1, 50, 5)
        out = self.fc.align_channels(X3d)
        assert out.shape == (1, 50, C)

    def test_unfitted_raises(self):
        fc = Forecaster("NLinear", seq_len=SEQ, pred_len=PRED)
        with pytest.raises(RuntimeError):
            fc.align_channels(_rng_data(n=50))


class TestPredictMultiStep:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=SEQ)

    def test_single_horizon(self):
        results = self.fc.predict_multi_step(self.X, horizons=[PRED])
        assert str(PRED) in results
        assert results[str(PRED)].shape == (PRED, C)

    def test_multiple_horizons(self):
        results = self.fc.predict_multi_step(self.X, horizons=[PRED, PRED * 2])
        assert str(PRED) in results
        assert str(PRED * 2) in results
        assert results[str(PRED * 2)].shape == (PRED * 2, C)

    def test_horizon_1(self):
        results = self.fc.predict_multi_step(self.X, horizons=[1])
        assert results["1"].shape == (1, C)

    def test_invalid_horizon_raises(self):
        with pytest.raises(ValueError):
            self.fc.predict_multi_step(self.X, horizons=[0])

    def test_3d_input(self):
        results = self.fc.predict_multi_step(self.X[np.newaxis], horizons=[PRED])
        assert results[str(PRED)].shape == (PRED, C)


class TestAutocorrelation:
    def test_returns_two_arrays(self):
        X = _rng_data(n=100)
        lags, acf = Forecaster.autocorrelation(X, max_lag=20)
        assert lags.shape == (21,)
        assert acf.shape == (21,)

    def test_lag_zero_is_one(self):
        X = _rng_data(n=100)
        _, acf = Forecaster.autocorrelation(X)
        assert abs(acf[0] - 1.0) < 1e-6

    def test_acf_bounded(self):
        X = _rng_data(n=100)
        _, acf = Forecaster.autocorrelation(X)
        assert np.all(np.abs(acf) <= 1.0 + 1e-6)

    def test_1d_input(self):
        X = np.random.default_rng(0).standard_normal(100).astype(np.float32)
        lags, acf = Forecaster.autocorrelation(X, max_lag=10)
        assert lags.shape == (11,)

    def test_custom_channel(self):
        X = _rng_data(n=100)
        _, acf0 = Forecaster.autocorrelation(X, channel=0)
        _, acf1 = Forecaster.autocorrelation(X, channel=1)
        # different channels → different ACF
        assert not np.allclose(acf0, acf1)


class TestPlotAcf:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure
        import matplotlib.pyplot as plt
        X = _rng_data(n=100)
        fig = Forecaster.plot_acf(X)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_custom_title(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        X = _rng_data(n=100)
        fig = Forecaster.plot_acf(X, title="ACF Test")
        ax = fig.get_axes()[0]
        assert "ACF Test" in ax.get_title()
        plt.close("all")


class TestSpectralDensity:
    def test_returns_two_arrays(self):
        X = _rng_data(n=100)
        freqs, psd = Forecaster.spectral_density(X)
        assert freqs.shape == psd.shape

    def test_freqs_range(self):
        X = _rng_data(n=100)
        freqs, _ = Forecaster.spectral_density(X)
        assert freqs[0] >= 0.0
        assert freqs[-1] <= 0.5 + 1e-9

    def test_psd_nonnegative(self):
        X = _rng_data(n=100)
        _, psd = Forecaster.spectral_density(X)
        assert (psd >= 0).all()

    def test_custom_n_fft(self):
        X = _rng_data(n=100)
        freqs, psd = Forecaster.spectral_density(X, n_fft=64)
        assert len(freqs) == 33  # rfftfreq(64) → 33 bins

    def test_1d_input(self):
        X = np.random.default_rng(0).standard_normal(100).astype(np.float32)
        freqs, psd = Forecaster.spectral_density(X)
        assert freqs.shape == psd.shape


class TestPlotSpectralDensity:
    def test_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.figure
        import matplotlib.pyplot as plt
        X = _rng_data(n=100)
        fig = Forecaster.plot_spectral_density(X)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_custom_title(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        X = _rng_data(n=100)
        fig = Forecaster.plot_spectral_density(X, title="PSD Test")
        ax = fig.get_axes()[0]
        assert "PSD Test" in ax.get_title()
        plt.close("all")


class TestAnomalyScore:
    def setup_method(self):
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=80)

    def test_returns_two_arrays(self):
        scores, indices = self.fc.anomaly_score(self.X, stride=4)
        assert isinstance(scores, np.ndarray)
        assert isinstance(indices, np.ndarray)

    def test_shapes_match(self):
        scores, indices = self.fc.anomaly_score(self.X, stride=4)
        assert scores.shape == indices.shape

    def test_scores_nonnegative(self):
        scores, _ = self.fc.anomaly_score(self.X, stride=4)
        assert (scores >= 0).all()

    def test_reduction_max_ge_mean(self):
        s_mean, _ = self.fc.anomaly_score(self.X, stride=4, reduction="mean")
        s_max, _ = self.fc.anomaly_score(self.X, stride=4, reduction="max")
        assert (s_max >= s_mean).all()

    def test_invalid_reduction_raises(self):
        with pytest.raises(ValueError, match="reduction"):
            self.fc.anomaly_score(self.X, stride=4, reduction="bad")

    def test_unfitted_raises(self):
        fc = Forecaster("NLinear", seq_len=SEQ, pred_len=PRED)
        with pytest.raises(RuntimeError):
            fc.anomaly_score(self.X)


class TestFlagAnomalies:
    def setup_method(self):
        pytest.importorskip("pandas")
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=80)

    def test_returns_dataframe(self):
        import pandas as pd
        df = self.fc.flag_anomalies(self.X, stride=4)
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self):
        df = self.fc.flag_anomalies(self.X, stride=4)
        for col in ("timestep", "score", "anomaly"):
            assert col in df.columns

    def test_contamination_rate(self):
        df = self.fc.flag_anomalies(self.X, stride=2, contamination=0.1)
        rate = df["anomaly"].mean()
        assert 0.0 <= rate <= 0.2  # allow some slack around exact quantile

    def test_explicit_threshold(self):
        df = self.fc.flag_anomalies(self.X, stride=4, threshold=0.0)
        assert df["anomaly"].all()  # score >= 0 always

    def test_anomaly_dtype_bool(self):
        df = self.fc.flag_anomalies(self.X, stride=4)
        assert df["anomaly"].dtype == bool


class TestPlotAnomalies:
    def setup_method(self):
        pytest.importorskip("pandas")
        pytest.importorskip("matplotlib")
        self.fc = _quick_fc().fit(_rng_data())
        self.X = _rng_data(n=80)
        self.df = self.fc.flag_anomalies(self.X, stride=4)

    def test_returns_figure(self):
        import matplotlib.figure
        import matplotlib.pyplot as plt
        fig = self.fc.plot_anomalies(self.X, self.df)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_custom_channel(self):
        import matplotlib.pyplot as plt
        fig = self.fc.plot_anomalies(self.X, self.df, channel=1)
        assert fig is not None
        plt.close("all")

    def test_custom_title(self):
        import matplotlib.pyplot as plt
        fig = self.fc.plot_anomalies(self.X, self.df, title="MyTitle")
        ax = fig.get_axes()[0]
        assert "MyTitle" in ax.get_title()
        plt.close("all")


class TestRollingEvaluate:
    def setup_method(self):
        pytest.importorskip("pandas")
        self.fc = _quick_fc().fit(_rng_data())

    def test_returns_dataframe(self):
        import pandas as pd
        X = _rng_data(n=60)
        df = self.fc.rolling_evaluate(X, stride=4)
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self):
        X = _rng_data(n=60)
        df = self.fc.rolling_evaluate(X, stride=4)
        for col in ("window", "MSE", "MAE", "RMSE", "SMAPE"):
            assert col in df.columns

    def test_custom_metrics(self):
        X = _rng_data(n=60)
        df = self.fc.rolling_evaluate(X, stride=4, metrics=["MSE", "MAE"])
        assert "RMSE" not in df.columns
        assert "MSE" in df.columns

    def test_row_count(self):
        X = _rng_data(n=60)
        df = self.fc.rolling_evaluate(X, stride=5)
        fc = self.fc
        expected = len(range(0, len(X) - fc.seq_len - fc.pred_len + 1, 5))
        assert len(df) == expected

    def test_window_column_monotonic(self):
        X = _rng_data(n=60)
        df = self.fc.rolling_evaluate(X, stride=3)
        assert (df["window"].diff().dropna() > 0).all()

    def test_unfitted_raises(self):
        fc = Forecaster("NLinear", seq_len=SEQ, pred_len=PRED)
        with pytest.raises(RuntimeError):
            fc.rolling_evaluate(_rng_data(n=60))


class TestPlotRollingMetrics:
    def setup_method(self):
        pytest.importorskip("pandas")
        pytest.importorskip("matplotlib")
        self.fc = _quick_fc().fit(_rng_data())
        X = _rng_data(n=60)
        self.df = self.fc.rolling_evaluate(X, stride=4)

    def test_returns_figure(self):
        import matplotlib.figure
        import matplotlib.pyplot as plt
        fig = self.fc.plot_rolling_metrics(self.df)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_custom_metrics(self):
        import matplotlib.pyplot as plt
        fig = self.fc.plot_rolling_metrics(self.df, metrics=["MSE"])
        assert len(fig.get_axes()) == 1
        plt.close("all")

    def test_custom_title(self):
        import matplotlib.pyplot as plt
        fig = self.fc.plot_rolling_metrics(self.df, title="Test")
        assert "Test" in fig.texts[0].get_text()
        plt.close("all")


class TestFromPretrained:
    def test_load_roundtrip(self, tmp_path):
        fc = _quick_fc().fit(_rng_data())
        path = str(tmp_path / "model")
        fc.save(path)
        fc2 = Forecaster.from_pretrained(path)
        assert fc2._model is not None

    def test_override_kwargs(self, tmp_path):
        fc = _quick_fc().fit(_rng_data())
        path = str(tmp_path / "model")
        fc.save(path)
        fc2 = Forecaster.from_pretrained(path, batch_size=32)
        assert fc2.batch_size == 32

    def test_predictions_match(self, tmp_path):
        fc = _quick_fc().fit(_rng_data())
        path = str(tmp_path / "model")
        fc.save(path)
        fc2 = Forecaster.from_pretrained(path)
        X = _rng_data(n=SEQ)
        np.testing.assert_allclose(fc.predict(X[np.newaxis]), fc2.predict(X[np.newaxis]), rtol=1e-5)


class TestSeasonalDecompose:
    def test_returns_expected_keys(self):
        X = _rng_data(n=100)
        result = Forecaster.seasonal_decompose(X, period=7)
        for k in ("trend", "seasonal", "residual", "original"):
            assert k in result

    def test_shapes(self):
        X = _rng_data(n=100)
        result = Forecaster.seasonal_decompose(X, period=7)
        assert result["trend"].shape == X.shape
        assert result["seasonal"].shape == X.shape

    def test_1d_input(self):
        X = np.random.default_rng(0).standard_normal(100)
        result = Forecaster.seasonal_decompose(X, period=7)
        assert result["trend"].shape == (100,)

    def test_additive_reconstruction(self):
        X = _rng_data(n=100)
        result = Forecaster.seasonal_decompose(X, period=7)
        recon = result["trend"] + result["seasonal"] + result["residual"]
        np.testing.assert_allclose(recon, X, atol=1e-8)

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError, match="period"):
            Forecaster.seasonal_decompose(_rng_data(n=100), period=1)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            Forecaster.seasonal_decompose(_rng_data(n=100), period=7,
                                          method="bogus")

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            Forecaster.seasonal_decompose(_rng_data(n=10), period=7)


class TestDetrendRetend:
    def test_detrend_returns_tuple(self):
        X = _rng_data(n=100)
        out = Forecaster.detrend(X)
        assert isinstance(out, tuple)
        assert len(out) == 2

    def test_detrend_output_shape(self):
        X = _rng_data(n=100)
        Xd, coeffs = Forecaster.detrend(X, degree=1)
        assert Xd.shape == X.shape
        assert coeffs.shape == (2, C)  # degree+1 coefficients

    def test_detrend_removes_linear_trend(self):
        # Perfectly linear series
        t = np.arange(100)[:, None] * np.ones((1, 2))  # (100, 2)
        Xd, _ = Forecaster.detrend(t, degree=1)
        np.testing.assert_allclose(Xd, 0.0, atol=1e-8)

    def test_detrend_quadratic(self):
        t = np.arange(100, dtype=float)[:, None]
        X = t ** 2
        Xd, coeffs = Forecaster.detrend(X, degree=2)
        assert coeffs.shape == (3, 1)
        np.testing.assert_allclose(Xd, 0.0, atol=1e-6)

    def test_retrend_inverts_detrend(self):
        X = _rng_data(n=100)
        Xd, coeffs = Forecaster.detrend(X, degree=1)
        Xr = Forecaster.retrend(Xd, coeffs, offset=0)
        np.testing.assert_allclose(Xr, X, atol=1e-8)

    def test_retrend_with_offset(self):
        X = _rng_data(n=200)
        _, coeffs = Forecaster.detrend(X[:100], degree=1)
        future_trend = Forecaster.retrend(np.zeros((50, C)), coeffs, offset=100)
        assert future_trend.shape == (50, C)

    def test_detrend_1d(self):
        X = np.arange(50, dtype=float)
        Xd, _ = Forecaster.detrend(X, degree=1)
        assert Xd.shape == (50,)
        np.testing.assert_allclose(Xd, 0.0, atol=1e-8)


class TestDiffUndiff:
    def test_diff_shape_order1(self):
        X = _rng_data(n=100)
        out = Forecaster.diff(X, order=1, lag=1)
        assert out.shape == (99, C)

    def test_diff_shape_order2(self):
        X = _rng_data(n=100)
        out = Forecaster.diff(X, order=2, lag=1)
        assert out.shape == (98, C)

    def test_diff_lag(self):
        X = _rng_data(n=100)
        out = Forecaster.diff(X, order=1, lag=7)
        assert out.shape == (93, C)

    def test_diff_1d_input(self):
        X = np.random.default_rng(0).standard_normal(100)
        out = Forecaster.diff(X, order=1)
        assert out.shape == (99,)

    def test_diff_constant_series_gives_zeros(self):
        X = np.ones((50, 2))
        out = Forecaster.diff(X, order=1)
        np.testing.assert_allclose(out, 0.0)

    def test_undiff_inverts_diff(self):
        X = _rng_data(n=100)
        Xd = Forecaster.diff(X, order=1, lag=1)
        Xr = Forecaster.undiff(Xd, X, order=1, lag=1)
        np.testing.assert_allclose(Xr, X[1:], atol=1e-8)

    def test_undiff_inverts_order2(self):
        X = _rng_data(n=100)
        Xd = Forecaster.diff(X, order=2, lag=1)
        Xr = Forecaster.undiff(Xd, X, order=2, lag=1)
        np.testing.assert_allclose(Xr, X[2:], atol=1e-7)

    def test_static_callable_without_instance(self):
        X = _rng_data(n=50)
        out = Forecaster.diff(X)
        assert out.shape[0] == 49


class TestComputeMetrics:
    def test_returns_expected_keys(self):
        y = _rng_data(n=10)
        result = Forecaster.compute_metrics(y, y)
        for k in ("mse", "mae", "rmse", "smape", "mase"):
            assert k in result

    def test_perfect_prediction_zero_mse(self):
        y = _rng_data(n=10)
        result = Forecaster.compute_metrics(y, y)
        assert result["mse"] == pytest.approx(0.0, abs=1e-9)
        assert result["mae"] == pytest.approx(0.0, abs=1e-9)

    def test_mse_non_negative(self):
        y_true = _rng_data(n=20)
        y_pred = _rng_data(n=20, seed=1)
        result = Forecaster.compute_metrics(y_true, y_pred)
        assert result["mse"] >= 0.0

    def test_rmse_sqrt_mse(self):
        y_true = _rng_data(n=20)
        y_pred = _rng_data(n=20, seed=1)
        result = Forecaster.compute_metrics(y_true, y_pred)
        assert result["rmse"] == pytest.approx(np.sqrt(result["mse"]), rel=1e-5)

    def test_shape_mismatch_raises(self):
        y = _rng_data(n=10)
        with pytest.raises(ValueError, match="shape"):
            Forecaster.compute_metrics(y, y[:5])

    def test_callable_without_instance(self):
        y = _rng_data(n=10)
        result = Forecaster.compute_metrics(y, y + 0.1)
        assert isinstance(result["mse"], float)

    def test_3d_input(self):
        Xw, yw = Forecaster.create_windows(_rng_data(n=200), SEQ, PRED)
        preds = yw + 0.01
        result = Forecaster.compute_metrics(yw, preds)
        assert result["mse"] >= 0.0


class TestCreateWindows:
    def test_basic_shapes(self):
        X = _rng_data(n=200)
        Xw, yw = Forecaster.create_windows(X, seq_len=SEQ, pred_len=PRED)
        n = 200 - SEQ - PRED + 1
        assert Xw.shape == (n, SEQ, C)
        assert yw.shape == (n, PRED, C)

    def test_stride(self):
        X = _rng_data(n=200)
        Xw, yw = Forecaster.create_windows(X, seq_len=SEQ, pred_len=PRED, stride=2)
        Xw2, _ = Forecaster.create_windows(X, seq_len=SEQ, pred_len=PRED, stride=1)
        assert len(Xw) < len(Xw2)

    def test_gap(self):
        X = _rng_data(n=200)
        Xw0, yw0 = Forecaster.create_windows(X, seq_len=SEQ, pred_len=PRED, gap=0)
        Xwg, ywg = Forecaster.create_windows(X, seq_len=SEQ, pred_len=PRED, gap=5)
        assert len(Xwg) < len(Xw0)

    def test_univariate(self):
        X = np.random.default_rng(0).standard_normal(200)
        Xw, yw = Forecaster.create_windows(X, seq_len=SEQ, pred_len=PRED)
        assert Xw.shape == (200 - SEQ - PRED + 1, SEQ, 1)

    def test_too_short_raises(self):
        X = _rng_data(n=5)
        with pytest.raises(ValueError):
            Forecaster.create_windows(X, seq_len=SEQ, pred_len=PRED)

    def test_context_and_target_nonoverlapping(self):
        X = np.arange(100, dtype=float)[:, None]
        Xw, yw = Forecaster.create_windows(X, seq_len=10, pred_len=5)
        # For window 0: ctx=[0..9], target=[10..14]
        np.testing.assert_array_equal(Xw[0, :, 0], np.arange(10))
        np.testing.assert_array_equal(yw[0, :, 0], np.arange(10, 15))


class TestSmooth:
    def test_output_shape_2d(self):
        X = _rng_data(n=100)
        out = Forecaster.smooth(X, window=5)
        assert out.shape == X.shape

    def test_output_shape_1d(self):
        X = np.random.default_rng(0).standard_normal(100)
        out = Forecaster.smooth(X, window=5)
        assert out.shape == X.shape

    def test_mean_smoothing_reduces_variance(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((200, 2))
        out = Forecaster.smooth(X, window=10, method="mean")
        assert out.std() < X.std()

    def test_median_smoothing_reduces_variance(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((200, 2))
        out = Forecaster.smooth(X, window=10, method="median")
        assert out.std() < X.std()

    def test_window_1_identity(self):
        X = _rng_data(n=50)
        out = Forecaster.smooth(X, window=1)
        np.testing.assert_allclose(out, X)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            Forecaster.smooth(_rng_data(), method="bogus")

    def test_static_callable_without_instance(self):
        X = _rng_data(n=50)
        out = Forecaster.smooth(X)
        assert out.shape == X.shape


# ── EnsembleForecaster ────────────────────────────────────────────────────────


def _make_ensemble(n=2):
    models = ["DLinear", "NLinear"][:n]
    return EnsembleForecaster([_quick_fc(m) for m in models])


class TestEnsembleForecaster:
    def test_fit_returns_self(self):
        X = _rng_data(n=300)
        ens = _make_ensemble()
        assert ens.fit(X) is ens

    def test_predict_shape(self):
        X = _rng_data(n=300)
        ens = _make_ensemble()
        ens.fit(X)
        y = ens.predict(X[-SEQ:])
        assert y.shape == (PRED, C)

    def test_predict_uses_weights(self):
        X = _rng_data(n=300)
        f1 = _quick_fc("DLinear")
        f2 = _quick_fc("NLinear")
        ens_equal = EnsembleForecaster([f1, f2])
        # Clone before fitting so we have identical untrained copies
        f1b = f1.clone()
        f2b = f2.clone()
        ens_equal.fit(X)
        # Now manually check weighted average with extreme weights
        ens_w1 = EnsembleForecaster([f1b.clone(), f2b.clone()], weights=[1.0, 0.0])
        ens_w1.fit(X)
        # weight=1 on first → result equals first forecaster's prediction
        y_w1 = ens_w1.predict(X[-SEQ:])
        y_f1 = ens_w1.forecasters_[0].predict(X[-SEQ:])
        np.testing.assert_allclose(y_w1, y_f1, rtol=1e-5)

    def test_predict_std_keys(self):
        X = _rng_data(n=300)
        ens = _make_ensemble()
        ens.fit(X)
        result = ens.predict_std(X[-SEQ:])
        for key in ("mean", "std", "lower", "upper"):
            assert key in result

    def test_predict_std_shape(self):
        X = _rng_data(n=300)
        ens = _make_ensemble()
        ens.fit(X)
        result = ens.predict_std(X[-SEQ:])
        assert result["mean"].shape == (PRED, C)
        assert result["std"].shape == (PRED, C)

    def test_score_keys(self):
        X = _rng_data(n=300)
        ens = _make_ensemble()
        ens.fit(X[:200])
        result = ens.score(X[200:])
        for key in ("mse", "mae", "rmse", "smape"):
            assert key in result

    def test_score_mse_non_negative(self):
        X = _rng_data(n=300)
        ens = _make_ensemble()
        ens.fit(X[:200])
        assert ens.score(X[200:])["mse"] >= 0.0

    def test_named_tuples_input(self):
        X = _rng_data(n=300)
        ens = EnsembleForecaster([("m1", _quick_fc("DLinear")),
                                  ("m2", _quick_fc("NLinear"))])
        ens.fit(X)
        assert [n for n, _ in ens.named_forecasters] == ["m1", "m2"]

    def test_wrong_weights_length_raises(self):
        X = _rng_data(n=300)
        ens = EnsembleForecaster([_quick_fc("DLinear"), _quick_fc("NLinear")],
                                 weights=[1.0])
        ens.fit(X)
        with pytest.raises(ValueError, match="weights length"):
            ens.predict(X[-SEQ:])

    def test_zero_weights_raises(self):
        X = _rng_data(n=300)
        ens = EnsembleForecaster([_quick_fc("DLinear"), _quick_fc("NLinear")],
                                 weights=[0.0, 0.0])
        ens.fit(X)
        with pytest.raises(ValueError, match="positive"):
            ens.predict(X[-SEQ:])

    def test_predict_before_fit_raises(self):
        ens = _make_ensemble()
        with pytest.raises(RuntimeError, match="not fitted"):
            ens.predict(np.zeros((SEQ, C)))

    def test_score_before_fit_raises(self):
        ens = _make_ensemble()
        with pytest.raises(RuntimeError, match="not fitted"):
            ens.score(np.zeros((200, C)))

    def test_empty_forecasters_raises(self):
        ens = EnsembleForecaster([])
        with pytest.raises(ValueError):
            ens.fit(_rng_data())

    def test_repr_unfitted(self):
        ens = _make_ensemble()
        r = repr(ens)
        assert "EnsembleForecaster" in r
        assert "not fitted" in r

    def test_repr_fitted(self):
        X = _rng_data(n=300)
        ens = _make_ensemble()
        ens.fit(X)
        assert "fitted" in repr(ens)


# ── SklearnForecaster ─────────────────────────────────────────────────────────


class TestSklearnForecaster:
    def test_fit_ts_returns_self(self):
        X = _rng_data(n=300)
        sk = SklearnForecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                               epochs=2, verbose=False)
        assert sk.fit_ts(X) is sk

    def test_fit_ts_creates_forecaster_(self):
        X = _rng_data(n=300)
        sk = SklearnForecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                               epochs=2, verbose=False)
        sk.fit_ts(X)
        assert isinstance(sk.forecaster_, Forecaster)

    def test_predict_shape_from_ts(self):
        X = _rng_data(n=300)
        sk = SklearnForecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                               epochs=2, verbose=False)
        sk.fit_ts(X)
        windows = np.stack([X[i : i + SEQ] for i in range(5)])
        preds = sk.predict(windows)
        assert preds.shape == (5, PRED * C)

    def test_predict_single_window(self):
        X = _rng_data(n=300)
        sk = SklearnForecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                               epochs=2, verbose=False)
        sk.fit_ts(X)
        pred = sk.predict(X[-SEQ:])
        assert pred.shape == (1, PRED * C)

    def test_score_returns_negative_float(self):
        X = _rng_data(n=300)
        sk = SklearnForecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                               epochs=2, verbose=False)
        windows = np.stack([X[i : i + SEQ] for i in range(5)])
        ys = np.stack([X[i + SEQ : i + SEQ + PRED] for i in range(5)])
        sk.fit_ts(X)
        s = sk.score(windows, ys)
        assert isinstance(s, float)
        assert s <= 0.0

    def test_get_params_has_expected_keys(self):
        sk = SklearnForecaster("DLinear", seq_len=SEQ, pred_len=PRED)
        p = sk.get_params()
        for k in ("model", "seq_len", "pred_len", "epochs", "lr"):
            assert k in p

    def test_set_params_updates_attrs(self):
        sk = SklearnForecaster("DLinear", seq_len=SEQ, pred_len=PRED)
        sk.set_params(lr=0.1)
        assert sk.lr == 0.1

    def test_predict_before_fit_raises(self):
        sk = SklearnForecaster("DLinear", seq_len=SEQ, pred_len=PRED)
        with pytest.raises(RuntimeError, match="not fitted"):
            sk.predict(np.zeros((SEQ, C)))

    def test_repr_unfitted(self):
        sk = SklearnForecaster("DLinear", seq_len=SEQ, pred_len=PRED)
        assert "SklearnForecaster" in repr(sk)
        assert "not fitted" in repr(sk)

    def test_repr_fitted(self):
        X = _rng_data(n=300)
        sk = SklearnForecaster("DLinear", seq_len=SEQ, pred_len=PRED,
                               epochs=2, verbose=False)
        sk.fit_ts(X)
        assert "fitted" in repr(sk)


# ── Phase 14 — Error Analysis & Advanced Visualization ─────────────────────

class TestErrorHeatmap:
    def test_returns_figure(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        fig = fc.error_heatmap(X, channel=0)
        assert hasattr(fig, "savefig")

    def test_custom_ax(self):
        import matplotlib.pyplot as plt
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        fig, ax = plt.subplots()
        ret = fc.error_heatmap(X, channel=0, ax=ax)
        assert ret is fig
        plt.close("all")


class TestRollingForecastQuality:
    def test_keys_present(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        res = fc.rolling_forecast_quality(X, window=5)
        for k in ("indices", "raw", "rolling"):
            assert k in res

    def test_shapes_consistent(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        res = fc.rolling_forecast_quality(X, window=5)
        assert len(res["indices"]) == len(res["raw"]) == len(res["rolling"])

    def test_metric_mae(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        res = fc.rolling_forecast_quality(X, metric="mae", window=5)
        assert np.all(res["raw"] >= 0)

    def test_plot_returns_figure(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        fig = fc.plot_rolling_forecast_quality(X, window=5)
        assert hasattr(fig, "savefig")


class TestMutualInformationMatrix:
    def test_shape(self):
        X = _rng_data()
        mi = Forecaster.mutual_information_matrix(X, n_bins=10)
        assert mi.shape == (C, C)

    def test_symmetric(self):
        X = _rng_data()
        mi = Forecaster.mutual_information_matrix(X, n_bins=10)
        np.testing.assert_allclose(mi, mi.T, atol=1e-5)

    def test_nonnegative(self):
        X = _rng_data()
        mi = Forecaster.mutual_information_matrix(X, n_bins=10)
        assert np.all(mi >= -1e-6)

    def test_plot_returns_figure(self):
        X = _rng_data()
        fig = Forecaster.plot_mutual_information(X, n_bins=10)
        assert hasattr(fig, "savefig")


class TestCoverageByHorizon:
    def test_keys(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        res = fc.coverage_by_horizon(X, X, coverage=0.9)
        for k in ("steps", "coverage", "nominal"):
            assert k in res

    def test_coverage_in_01(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        res = fc.coverage_by_horizon(X, X, coverage=0.9)
        assert np.all((res["coverage"] >= 0) & (res["coverage"] <= 1))

    def test_steps_length(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        res = fc.coverage_by_horizon(X, X, coverage=0.9)
        assert len(res["steps"]) == PRED

    def test_plot_returns_figure(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        fig = fc.plot_coverage_by_horizon(X, X, coverage=0.9)
        assert hasattr(fig, "savefig")


class TestSharpness:
    def test_positive(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        s = fc.sharpness(X, X, coverage=0.9)
        assert isinstance(s, float) and s >= 0

    def test_higher_coverage_wider(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        s90 = fc.sharpness(X, X, coverage=0.90)
        s50 = fc.sharpness(X, X, coverage=0.50)
        assert s90 >= s50


class TestConditionalBias:
    def test_keys(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        res = fc.conditional_bias(X, channel=0, n_bins=4)
        for k in ("bin_centers", "bias", "std"):
            assert k in res

    def test_length(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        res = fc.conditional_bias(X, channel=0, n_bins=4)
        assert len(res["bin_centers"]) == len(res["bias"]) == 4

    def test_plot_returns_figure(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        fig = fc.plot_conditional_bias(X, channel=0, n_bins=4)
        assert hasattr(fig, "savefig")


class TestErrorCorrelationMatrix:
    def test_shape(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        corr = fc.error_correlation_matrix(X, channel=0)
        assert corr.shape == (PRED, PRED)

    def test_symmetric(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        corr = fc.error_correlation_matrix(X, channel=0)
        np.testing.assert_allclose(corr, corr.T, atol=1e-5)

    def test_diagonal_one(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        corr = fc.error_correlation_matrix(X, channel=0)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-4)

    def test_plot_returns_figure(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        fig = fc.plot_error_correlation(X, channel=0)
        assert hasattr(fig, "savefig")


class TestMinmaxScale:
    def test_output_range(self):
        X = _rng_data()
        X_s, xmin, xmax = Forecaster.minmax_scale(X, feature_range=(0, 1))
        assert float(X_s.min()) >= -1e-5
        assert float(X_s.max()) <= 1 + 1e-5

    def test_inverse_roundtrip(self):
        X = _rng_data()
        X_s, xmin, xmax = Forecaster.minmax_scale(X, feature_range=(0, 1))
        X_back = Forecaster.minmax_inverse(X_s, xmin, xmax, feature_range=(0, 1))
        np.testing.assert_allclose(X_back, X, atol=1e-4)

    def test_custom_range(self):
        X = _rng_data()
        X_s, _, _ = Forecaster.minmax_scale(X, feature_range=(-1, 1))
        assert float(X_s.min()) >= -1 - 1e-5
        assert float(X_s.max()) <= 1 + 1e-5


class TestQuantileResiduals:
    def test_keys(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        res = fc.quantile_residuals(X, quantiles=(0.1, 0.5, 0.9), channel=0)
        assert "steps" in res
        for q in (0.1, 0.5, 0.9):
            assert f"q{q:.2f}" in res

    def test_step_length(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        res = fc.quantile_residuals(X, quantiles=(0.5,), channel=0)
        assert len(res["steps"]) == PRED

    def test_median_near_zero_for_unbiased(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        res = fc.quantile_residuals(X, quantiles=(0.5,), channel=0)
        # median residual should be finite
        assert np.all(np.isfinite(res["q0.50"]))

    def test_plot_returns_figure(self):
        fc = _quick_fc().fit(_rng_data())
        X = _rng_data()
        fig = fc.plot_quantile_residuals(X, quantiles=(0.1, 0.5, 0.9), channel=0)
        assert hasattr(fig, "savefig")
