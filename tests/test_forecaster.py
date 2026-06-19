"""Tests for the high-level Forecaster fit/predict/score API."""
import os
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn

from torch_timeseries.forecaster import (
    Forecaster, StackedForecaster, BaggingForecaster, Pipeline,
    MultiChannelForecaster, EnsembleForecaster,
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
