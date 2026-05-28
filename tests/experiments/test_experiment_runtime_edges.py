import types

import pytest


def test_imputation_wandb_config_uses_result_related_configs(monkeypatch):
    from torch_timeseries.experiments.imputation import ImputationExp

    captured = {}

    class Runs:
        def __iter__(self):
            return iter(())

        def __getitem__(self, index):
            raise IndexError

    class Api:
        def runs(self, path, filters):
            captured["path"] = path
            captured["filters"] = filters
            return Runs()

    wandb = types.SimpleNamespace(
        Api=lambda: Api(),
        init=lambda **kwargs: types.SimpleNamespace(),
        config=types.SimpleNamespace(update=lambda value: None),
    )
    monkeypatch.setattr("torch_timeseries.experiments.imputation.wandb", wandb)

    exp = ImputationExp(model_type="DLinear", dataset_type="ETTh1")

    assert exp.config_wandb("project", "name") is exp
    assert "config.model_type" in captured["filters"]


def test_anomaly_validation_resets_metrics_before_update(monkeypatch):
    from torch_timeseries.experiments.anomaly_detection import AnomalyDetectionExp

    class Metrics:
        def __init__(self):
            self.reset_called = False

        def reset(self):
            self.reset_called = True

        def update(self, *_):
            assert self.reset_called

        def items(self):
            metric = types.SimpleNamespace(compute=lambda: 0.0)
            return {"mse": metric, "mae": metric}.items()

    class Model:
        def eval(self):
            pass

    exp = AnomalyDetectionExp()
    exp.model = Model()
    exp.metrics = Metrics()

    class Loader:
        dataset = []

        def __iter__(self):
            return iter(())

    exp.val_loader = Loader()
    exp.run_save_dir = "."
    monkeypatch.setattr(exp, "_run_print", lambda *args, **kwargs: None)

    exp._val()

    assert exp.metrics.reset_called


def test_forecast_exp_uses_forecast_data_module(monkeypatch):
    """ForecastExp._init_data_loader must build a ForecastDataModule, not ETTHLoader."""
    from torch_timeseries.experiments.forecast import ForecastExp
    from torch_timeseries.dataloader.v2 import ForecastDataModule

    built = {}
    _orig_init = ForecastDataModule.__init__

    def _spy_init(self, *args, **kwargs):
        built["called"] = True
        _orig_init(self, *args, **kwargs)

    monkeypatch.setattr(ForecastDataModule, "__init__", _spy_init)

    class FakeForecast(ForecastExp):
        model_type = "DLinear"
        dataset_type = "ETTh1"

        def _init_dataset(self):
            import numpy as np
            import pandas as pd
            from torch_timeseries.core import TimeSeriesDataset, Freq

            class _DS(TimeSeriesDataset):
                name = "ETTh1"; num_features = 7; freq = Freq.hours
                def download(self): pass
                def _load(self):
                    n = 1000
                    rng = np.random.default_rng(0)
                    self.df = pd.DataFrame(
                        {"date": pd.date_range("2020-01-01", periods=n, freq="h"),
                         **{f"c{i}": rng.normal(size=n) for i in range(7)}}
                    )
                    self.dates = self.df[["date"]]
                    self.data  = self.df.drop("date", axis=1).values
                    self.length = n

            self.dataset = _DS(root="/tmp")

    exp = FakeForecast()
    exp._init_data_loader()

    assert built.get("called"), "ForecastDataModule was not constructed"
    assert hasattr(exp, "train_loader")
    assert hasattr(exp, "val_loader")
    assert hasattr(exp, "test_loader")
