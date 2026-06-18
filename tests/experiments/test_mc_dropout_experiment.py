"""End-to-end smoke tests for MCDropoutForecast experiment."""
import types

import numpy as np
import pandas as pd
import pytest
import torch

from torch_timeseries.experiments.MCDropoutForecaster import MCDropoutForecast
from torch_timeseries.model.MCDropoutForecaster import MCDropoutForecaster


# ── minimal in-memory dataset ─────────────────────────────────────────────────


class _ToyDataset:
    """Minimal dataset that satisfies ForecastDataModule contract."""
    name = "__toy_mc__"
    freq = "h"

    def __init__(self, T=200, C=3):
        rng = np.random.default_rng(0)
        self.data = rng.standard_normal((T, C)).astype(np.float32)
        self.num_features = C
        self.length = T
        self.df = pd.DataFrame(
            {"date": pd.date_range("2020-01-01", periods=T, freq="h"),
             **{f"f{i}": self.data[:, i] for i in range(C)}}
        )
        self.dates = pd.DataFrame({"date": self.df["date"]})
        self.time_index = np.arange(T)

    def __len__(self):
        return self.length


def _make_exp(tmp_path, T=200, C=3, windows=12, pred_len=4,
              num_samples=5, epochs=1):
    """Create an MCDropoutForecast with an injected toy dataset."""
    toy = _ToyDataset(T=T, C=C)

    exp = MCDropoutForecast(
        dataset_type="__toy_mc__",
        windows=windows,
        pred_len=pred_len,
        horizon=1,
        num_samples=num_samples,
        d_model=32,
        n_heads=2,
        e_layers=1,
        d_ff=64,
        dropout=0.1,
        device="cpu",
        batch_size=8,
        epochs=epochs,
        patience=100,
        save_dir=str(tmp_path),
    )
    # Inject the toy dataset so _init_data_loader() skips download.
    exp._init_dataset = types.MethodType(
        lambda s: setattr(s, "dataset", toy), exp
    )
    return exp


# ── construction / setup ─────────────────────────────────────────────────────


class TestMCDropoutForecastSetup:
    def test_init_model_creates_mcdropout_instance(self, tmp_path):
        exp = _make_exp(tmp_path)
        exp._setup()
        exp._init_model()
        assert isinstance(exp.model, MCDropoutForecaster)

    def test_model_num_samples_matches_config(self, tmp_path):
        exp = _make_exp(tmp_path, num_samples=7)
        exp._setup()
        exp._init_model()
        assert exp.model.num_samples == 7

    def test_prob_metrics_initialized(self, tmp_path):
        exp = _make_exp(tmp_path)
        exp._setup()
        exp._init_model()
        metric_names = set(exp.metrics.keys())
        for expected in ("crps", "crps_sum", "picp", "qice", "mse", "mae", "rmse"):
            assert expected in metric_names, f"missing metric: {expected}"

    def test_monitor_metric_is_crps(self, tmp_path):
        exp = _make_exp(tmp_path)
        assert exp.monitor_metric == "crps"

    def test_loaders_created(self, tmp_path):
        exp = _make_exp(tmp_path)
        exp._setup()
        assert exp.train_loader is not None
        assert exp.val_loader is not None
        assert exp.test_loader is not None


# ── single-batch forward ──────────────────────────────────────────────────────


class TestMCDropoutForecastBatch:
    def test_train_batch_returns_scalar_loss(self, tmp_path):
        exp = _make_exp(tmp_path)
        exp._setup()
        exp._init_model()
        exp.model_optim = torch.optim.Adam(exp.model.parameters())
        batch = next(iter(exp.train_loader))
        loss = exp._process_train_batch(batch)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_train_batch_loss_backward(self, tmp_path):
        exp = _make_exp(tmp_path)
        exp._setup()
        exp._init_model()
        exp.model_optim = torch.optim.Adam(exp.model.parameters())
        batch = next(iter(exp.train_loader))
        loss = exp._process_train_batch(batch)
        loss.backward()  # should not raise

    def test_val_batch_preds_shape(self, tmp_path):
        exp = _make_exp(tmp_path, num_samples=5)
        exp._setup()
        exp._init_model()
        batch = next(iter(exp.val_loader))
        preds, truths = exp._process_val_batch(batch)
        B = batch.x.size(0)
        C = exp.dataset.num_features
        assert preds.shape == (B, 4, C, 5)   # (B, pred_len, N, S)
        assert truths.shape == (B, 4, C)      # (B, pred_len, N)

    def test_val_batch_preds_finite(self, tmp_path):
        exp = _make_exp(tmp_path)
        exp._setup()
        exp._init_model()
        batch = next(iter(exp.val_loader))
        preds, truths = exp._process_val_batch(batch)
        assert torch.isfinite(preds).all()
        assert torch.isfinite(truths).all()


# ── full experiment run ───────────────────────────────────────────────────────


class TestMCDropoutForecastRun:
    def test_run_returns_all_prob_metrics(self, tmp_path):
        exp = _make_exp(tmp_path, T=200, epochs=1)
        result = exp.run(seed=0)
        assert isinstance(result, dict)
        for key in ("crps", "crps_sum", "picp", "qice", "mse", "mae", "rmse"):
            assert key in result, f"missing key: {key}"
            assert np.isfinite(result[key]), f"{key} = {result[key]}"

    def test_run_crps_non_negative(self, tmp_path):
        exp = _make_exp(tmp_path, T=200, epochs=1)
        result = exp.run(seed=42)
        assert result["crps"] >= 0

    def test_run_picp_in_unit_interval(self, tmp_path):
        exp = _make_exp(tmp_path, T=200, epochs=1)
        result = exp.run(seed=7)
        assert 0 <= result["picp"] <= 1

    def test_run_mse_positive(self, tmp_path):
        exp = _make_exp(tmp_path, T=200, epochs=1)
        result = exp.run(seed=3)
        assert result["mse"] >= 0
        assert result["mae"] >= 0
        assert result["rmse"] >= 0
