import pytest


def test_forecast_dlinear_config_split_accepts_flat_experiment_kwargs():
    from torch_timeseries.experiments.configs import split_experiment_config

    task_cfg, model_cfg, runtime_cfg = split_experiment_config(
        model="DLinear",
        task="Forecast",
        kwargs={
            "windows": 96,
            "pred_len": 336,
            "horizon": 1,
            "input_columns": [0, 1],
            "target_columns": [2],
            "individual": True,
            "batch_size": 16,
            "epochs": 2,
            "device": "cpu",
            "save_dir": "./tmp-results",
        },
    )

    assert task_cfg.windows == 96
    assert task_cfg.pred_len == 336
    assert task_cfg.horizon == 1
    assert task_cfg.input_columns == [0, 1]
    assert task_cfg.target_columns == [2]
    assert model_cfg.individual is True
    assert runtime_cfg.batch_size == 16
    assert runtime_cfg.epochs == 2
    assert runtime_cfg.device == "cpu"
    assert runtime_cfg.save_dir == "./tmp-results"


def test_forecast_crossformer_config_split_accepts_flat_experiment_kwargs():
    from torch_timeseries.experiments.configs import split_experiment_config

    task_cfg, model_cfg, runtime_cfg = split_experiment_config(
        model="Crossformer",
        task="Forecast",
        kwargs={
            "windows": 48,
            "pred_len": 24,
            "seg_len": 4,
            "win_size": 3,
            "factor": 8,
            "d_model": 32,
            "d_ff": 64,
            "n_heads": 2,
            "e_layers": 1,
            "dropout": 0.1,
            "baseline": True,
            "epochs": 1,
            "save_dir": "./tmp-results",
        },
    )

    assert task_cfg.windows == 48
    assert task_cfg.pred_len == 24
    assert model_cfg.seg_len == 4
    assert model_cfg.win_size == 3
    assert model_cfg.factor == 8
    assert model_cfg.d_model == 32
    assert model_cfg.d_ff == 64
    assert model_cfg.n_heads == 2
    assert model_cfg.e_layers == 1
    assert model_cfg.dropout == 0.1
    assert model_cfg.baseline is True
    assert runtime_cfg.epochs == 1
    assert runtime_cfg.save_dir == "./tmp-results"


def test_forecast_dlinear_config_rejects_irrelevant_model_kwargs():
    from torch_timeseries.experiments.configs import split_experiment_config

    with pytest.raises(TypeError, match="Unknown or irrelevant configuration keys: d_model"):
        split_experiment_config(
            model="DLinear",
            task="Forecast",
            kwargs={"windows": 96, "pred_len": 96, "d_model": 512},
        )


def test_forecast_crossformer_config_rejects_irrelevant_model_kwargs():
    from torch_timeseries.experiments.configs import split_experiment_config

    with pytest.raises(
        TypeError,
        match="Unknown or irrelevant configuration keys: individual",
    ):
        split_experiment_config(
            model="Crossformer",
            task="Forecast",
            kwargs={"windows": 96, "pred_len": 96, "individual": True},
        )


def test_forecast_config_validates_shape_controls():
    from torch_timeseries.experiments.configs import split_experiment_config

    with pytest.raises(ValueError, match="pred_len must be positive"):
        split_experiment_config(
            model="DLinear",
            task="Forecast",
            kwargs={"windows": 96, "pred_len": 0},
        )


def test_dlinear_forecast_engine_uses_tsbatch_and_returns_metrics(monkeypatch, tmp_path):
    import numpy as np
    import pandas as pd
    import torch
    from torch_timeseries.core import Freq, TimeSeriesDataset
    from torch_timeseries.dataloader.v2 import SplitConfig, WindowConfig
    from torch_timeseries.experiments.configs import (
        DLinearConfig,
        RuntimeConfig,
    )
    from torch_timeseries.experiments.engine import DLinearForecastEngine

    class TinyForecastDataset(TimeSeriesDataset):
        name = "TinyForecast"
        num_features = 3
        freq = Freq.hours

        def download(self):
            pass

        def _load(self):
            n = 160
            rng = np.random.default_rng(7)
            self.df = pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=n, freq="h"),
                    **{f"c{i}": rng.normal(size=n) for i in range(3)},
                }
            )
            self.dates = self.df[["date"]]
            self.data = self.df.drop("date", axis=1).values
            self.length = n

    seen = {"batch_type": None}

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return x[:, -4:, :] + self.bias

    class TinyEngine(DLinearForecastEngine):
        def _init_dataset(self):
            self.dataset = TinyForecastDataset(root=str(tmp_path / "data"))

        def _build_model(self):
            self.model = TinyModel().to(self.runtime.device)

        def _process_batch(self, batch):
            seen["batch_type"] = type(batch).__name__
            return super()._process_batch(batch)

    engine = TinyEngine(
        model_name="DLinear",
        dataset_name="TinyForecast",
        window_config=WindowConfig(window=8, steps=4),
        split_config=SplitConfig(train=0.6, test=0.2),
        model_config=DLinearConfig(),
        runtime_config=RuntimeConfig(
            epochs=1,
            batch_size=8,
            save_dir=str(tmp_path / "runs"),
        ),
    )

    result = engine.run(seed=1)

    assert seen["batch_type"] == "TSBatch"
    assert set(result) == {"mse", "mae"}
    assert engine.datamodule.num_target_features == 3


def test_crossformer_forecast_engine_builds_model_from_typed_config(monkeypatch, tmp_path):
    import numpy as np
    import pandas as pd
    import torch
    from torch_timeseries.core import Freq, TimeSeriesDataset
    from torch_timeseries.dataloader.v2 import SplitConfig, WindowConfig
    from torch_timeseries.experiments.configs import (
        CrossformerConfig,
        RuntimeConfig,
    )
    from torch_timeseries.experiments.engine import CrossformerForecastEngine

    class TinyForecastDataset(TimeSeriesDataset):
        name = "TinyForecast"
        num_features = 2
        freq = Freq.hours

        def download(self):
            pass

        def _load(self):
            n = 120
            rng = np.random.default_rng(11)
            self.df = pd.DataFrame(
                {
                    "date": pd.date_range("2021-01-01", periods=n, freq="h"),
                    **{f"c{i}": rng.normal(size=n) for i in range(2)},
                }
            )
            self.dates = self.df[["date"]]
            self.data = self.df.drop("date", axis=1).values
            self.length = n

    captured = {}

    class TinyCrossformer(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)
            self.bias = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return x[:, -3:, :] + self.bias

    monkeypatch.setattr("torch_timeseries.experiments.engine.Crossformer", TinyCrossformer)

    class TinyEngine(CrossformerForecastEngine):
        def _init_dataset(self):
            self.dataset = TinyForecastDataset(root=str(tmp_path / "data"))

    engine = TinyEngine(
        model_name="Crossformer",
        dataset_name="TinyForecast",
        window_config=WindowConfig(window=6, steps=3),
        split_config=SplitConfig(train=0.6, test=0.2),
        model_config=CrossformerConfig(
            seg_len=2,
            win_size=3,
            factor=4,
            d_model=8,
            d_ff=16,
            n_heads=1,
            e_layers=1,
            dropout=0.0,
            baseline=True,
        ),
        runtime_config=RuntimeConfig(
            epochs=1,
            batch_size=8,
            save_dir=str(tmp_path / "runs"),
        ),
    )

    result = engine.run(seed=1)

    assert set(result) == {"mse", "mae"}
    assert captured["data_dim"] == 2
    assert captured["in_len"] == 6
    assert captured["out_len"] == 3
    assert captured["seg_len"] == 2
    assert captured["win_size"] == 3
    assert captured["factor"] == 4
    assert captured["d_model"] == 8
    assert captured["d_ff"] == 16
    assert captured["n_heads"] == 1
    assert captured["e_layers"] == 1
    assert captured["dropout"] == 0.0
    assert captured["baseline"] is True
    assert captured["device"] == "cpu"


def test_experiment_constructor_accepts_flat_dlinear_forecast_config(monkeypatch, tmp_path):
    from torch_timeseries.experiment import Experiment

    captured = {}

    class FakeEngine:
        def __init__(
            self,
            model_name,
            dataset_name,
            task_config,
            model_config,
            runtime_config,
        ):
            captured["model_name"] = model_name
            captured["dataset_name"] = dataset_name
            captured["task_config"] = task_config
            captured["model_config"] = model_config
            captured["runtime_config"] = runtime_config
            self.model = None

        def run(self, seed):
            captured["seed"] = seed
            return {"mse": 0.12, "mae": 0.08}

        def hparams(self):
            return {
                "windows": captured["task_config"].windows,
                "individual": captured["model_config"].individual,
            }

        def num_parameters(self):
            return 5

    monkeypatch.setattr("torch_timeseries.experiment.DLinearForecastEngine", FakeEngine)

    results = Experiment(
        model="DLinear",
        task="Forecast",
        dataset="ETTh1",
        windows=24,
        pred_len=12,
        individual=True,
        epochs=1,
        save_dir=str(tmp_path),
    ).run(seeds=[7])

    assert captured["model_name"] == "DLinear"
    assert captured["dataset_name"] == "ETTh1"
    assert captured["task_config"].windows == 24
    assert captured["task_config"].pred_len == 12
    assert captured["model_config"].individual is True
    assert captured["runtime_config"].epochs == 1
    assert captured["seed"] == 7
    assert results[0].metrics == {"mse": 0.12, "mae": 0.08}
    assert results[0].hparams == {"windows": 24, "individual": True}
    assert results[0].num_params == 5


def test_experiment_constructor_saves_dlinear_forecast_local_result(monkeypatch, tmp_path):
    from torch_timeseries.experiment import Experiment

    class FakeEngine:
        def __init__(self, model_name, dataset_name, task_config, model_config, runtime_config):
            self.best_checkpoint_filepath = str(tmp_path / "source_best_model.pth")
            self.history = {"train_loss": [0.1], "val": [{"mse": 0.2}]}
            self._hparams = {
                "windows": task_config.windows,
                "pred_len": task_config.pred_len,
                "lr": runtime_config.lr,
                "save_dir": runtime_config.save_dir,
            }

        def run(self, seed):
            with open(self.best_checkpoint_filepath, "wb") as f:
                f.write(b"checkpoint")
            return {"mse": 0.2}

        def hparams(self):
            return dict(self._hparams)

        def num_parameters(self):
            return 9

    monkeypatch.setattr("torch_timeseries.experiment.DLinearForecastEngine", FakeEngine)

    results = Experiment(
        model="DLinear",
        task="Forecast",
        dataset="ETTh1",
        windows=8,
        pred_len=4,
        save_dir=str(tmp_path),
    ).run(seeds=[1])

    files = list((tmp_path / "records").rglob("seed1.json"))
    assert len(files) == 1
    assert results[0].metrics["mse"] == 0.2
    assert results[0].config_hash
    assert results[0].run_id == f"seed1-{results[0].config_hash}"
    assert results[0].run_config["hparams"]["windows"] == 8
    assert results[0].run_config["hparams"]["pred_len"] == 4
    assert results[0].run_config["hparams"]["lr"] == 0.0001
    assert results[0].run_config["hparams"]["batch_size"] == 32
    assert "save_dir" not in results[0].run_config["hparams"]
    assert results[0].artifacts["model"].endswith("best_model.pth")
    assert results[0].config_hash in results[0].artifacts["model"]
    assert (tmp_path / results[0].artifacts["model"]).exists()


def test_experiment_constructor_accepts_flat_crossformer_forecast_config(monkeypatch, tmp_path):
    from torch_timeseries.experiment import Experiment

    captured = {}

    class FakeEngine:
        def __init__(
            self,
            model_name,
            dataset_name,
            task_config,
            model_config,
            runtime_config,
        ):
            captured["model_name"] = model_name
            captured["dataset_name"] = dataset_name
            captured["task_config"] = task_config
            captured["model_config"] = model_config
            captured["runtime_config"] = runtime_config

        def run(self, seed):
            captured["seed"] = seed
            return {"mse": 0.22, "mae": 0.11}

        def hparams(self):
            return {
                "windows": captured["task_config"].windows,
                "seg_len": captured["model_config"].seg_len,
            }

        def num_parameters(self):
            return 13

    monkeypatch.setattr("torch_timeseries.experiment.CrossformerForecastEngine", FakeEngine)

    results = Experiment(
        model="Crossformer",
        task="Forecast",
        dataset="ETTh1",
        windows=24,
        pred_len=12,
        seg_len=3,
        epochs=1,
        save_dir=str(tmp_path),
    ).run(seeds=[9])

    assert captured["model_name"] == "Crossformer"
    assert captured["dataset_name"] == "ETTh1"
    assert captured["task_config"].windows == 24
    assert captured["task_config"].pred_len == 12
    assert captured["model_config"].seg_len == 3
    assert captured["runtime_config"].epochs == 1
    assert captured["seed"] == 9
    assert results[0].metrics == {"mse": 0.22, "mae": 0.11}
    assert results[0].hparams == {"windows": 24, "seg_len": 3}
    assert results[0].num_params == 13


def test_dlinear_forecast_class_is_compatibility_shim(monkeypatch):
    from torch_timeseries.experiments.DLinear import DLinearForecast

    captured = {}

    def fake_run_engine(self, seed):
        captured["windows"] = self.windows
        captured["pred_len"] = self.pred_len
        captured["individual"] = self.individual
        captured["seed"] = seed
        return {"mse": 0.33}

    monkeypatch.setattr(DLinearForecast, "_run_engine_compat", fake_run_engine)

    exp = DLinearForecast(windows=16, pred_len=8, individual=True)
    result = exp.run(seed=5)

    assert result == {"mse": 0.33}
    assert captured == {
        "windows": 16,
        "pred_len": 8,
        "individual": True,
        "seed": 5,
    }


def test_crossformer_forecast_class_is_compatibility_shim(monkeypatch):
    from torch_timeseries.experiments.Crossformer import CrossformerForecast

    captured = {}

    def fake_run_engine(self, seed):
        captured["windows"] = self.windows
        captured["pred_len"] = self.pred_len
        captured["seg_len"] = self.seg_len
        captured["seed"] = seed
        return {"mse": 0.44}

    monkeypatch.setattr(CrossformerForecast, "_run_engine_compat", fake_run_engine)

    exp = CrossformerForecast(windows=18, pred_len=6, seg_len=3)
    result = exp.run(seed=7)

    assert result == {"mse": 0.44}
    assert captured == {
        "windows": 18,
        "pred_len": 6,
        "seg_len": 3,
        "seed": 7,
    }


def test_readme_teaches_constructor_first_experiment_api():
    text = open("README.md", encoding="utf-8").read()
    assert "Experiment(\n    model=\"DLinear\"" in text
    assert "save_dir=\"./results\"" in text


def test_forecast_engine_train_epoch_uses_tqdm_progress_bar(monkeypatch, tmp_path):
    import torch
    import torch_timeseries.experiments.engine as engine_module
    from torch_timeseries.dataloader.v2 import SplitConfig, WindowConfig
    from torch_timeseries.experiments.configs import (
        DLinearConfig,
        RuntimeConfig,
    )
    from torch_timeseries.experiments.engine import DLinearForecastEngine

    calls = {"totals": [], "updates": [], "postfixes": []}

    class FakeTqdm:
        def __init__(self, total):
            calls["totals"].append(total)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, amount):
            calls["updates"].append(amount)

        def set_postfix(self, **kwargs):
            calls["postfixes"].append(kwargs)

    class TinyBatch:
        def __init__(self):
            self.x = torch.ones(2, 1)
            self.y = torch.zeros(2, 1)
            self.y_raw = self.y

        def to(self, device):
            return self

    class TinyLoader:
        dataset = [None, None, None, None]

        def __iter__(self):
            return iter([TinyBatch(), TinyBatch()])

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return x + self.bias

    monkeypatch.setattr(engine_module, "tqdm", FakeTqdm, raising=False)

    engine = DLinearForecastEngine(
        model_name="DLinear",
        dataset_name="TinyForecast",
        window_config=WindowConfig(window=2, steps=1),
        model_config=DLinearConfig(),
        runtime_config=RuntimeConfig(
            epochs=1,
            batch_size=2,
            save_dir=str(tmp_path / "runs"),
        ),
    )
    engine.model = TinyModel()
    engine.train_loader = TinyLoader()
    engine.optimizer = torch.optim.SGD(engine.model.parameters(), lr=0.1)
    engine.loss_func = torch.nn.MSELoss()

    engine._train_epoch()

    assert calls["totals"] == [4]
    assert calls["updates"] == [2, 2]
    assert calls["postfixes"]
    assert {"loss", "lr", "epoch"}.issubset(calls["postfixes"][-1])


def test_forecast_engine_reports_train_loss_and_val_history_each_epoch(capsys, tmp_path):
    from torch_timeseries.dataloader.v2 import SplitConfig, WindowConfig
    from torch_timeseries.experiments.configs import (
        DLinearConfig,
        RuntimeConfig,
    )
    from torch_timeseries.experiments.engine import ForecastEngine

    class NoopScheduler:
        def step(self):
            pass

    class NoopEarlyStopping:
        early_stop = False

        def __call__(self, score, model):
            pass

    class TinyEngine(ForecastEngine):
        def setup(self):
            self.model = object()
            self.val_loader = object()
            self.test_loader = object()
            self.scheduler = NoopScheduler()
            self.early_stopper = NoopEarlyStopping()
            self.best_checkpoint_filepath = str(tmp_path / "missing.pt")

        def _train_epoch(self):
            base = self.current_epoch + 1
            return [float(base), float(base + 2)]

        def _evaluate(self, loader):
            if loader is self.val_loader:
                return {
                    "mse": self.current_epoch + 0.5,
                    "mae": 0.25,
                }
            return {"mse": 9.0, "mae": 8.0}

    engine = TinyEngine(
        model_name="DLinear",
        dataset_name="TinyForecast",
        window_config=WindowConfig(window=2, steps=1),
        model_config=DLinearConfig(),
        runtime_config=RuntimeConfig(
            epochs=2,
            save_dir=str(tmp_path / "runs"),
        ),
    )

    result = engine.run(seed=1)

    out = capsys.readouterr().out
    assert result == {"mse": 9.0, "mae": 8.0}
    assert "Epoch 1/2" in out
    assert "train_loss=2" in out
    assert "val_mae=0.25" in out
    assert "val_mse=0.5" in out
    assert "Epoch 2/2" in out
    assert "train_loss=3" in out
    assert "val_mse=1.5" in out
    assert engine.history == {
        "train_loss": [2.0, 3.0],
        "val": [
            {"mse": 0.5, "mae": 0.25},
            {"mse": 1.5, "mae": 0.25},
        ],
    }


def test_experiment_engine_path_stores_history_in_run_result(monkeypatch, tmp_path):
    from torch_timeseries.experiment import Experiment

    class FakeEngine:
        def __init__(self, *args, **kwargs):
            self.history = None

        def run(self, seed):
            self.history = {
                "train_loss": [0.8],
                "val": [{"mse": 0.4, "mae": 0.3}],
            }
            return {"mse": 0.5, "mae": 0.35}

        def hparams(self):
            return {"windows": 8}

        def num_parameters(self):
            return 7

    monkeypatch.setattr("torch_timeseries.experiment.DLinearForecastEngine", FakeEngine)

    results = Experiment(
        model="DLinear",
        task="Forecast",
        dataset="ETTh1",
        windows=8,
        pred_len=4,
        save_dir=str(tmp_path),
    ).run(seeds=[1])

    assert results[0].history == {
        "train_loss": [0.8],
        "val": [{"mse": 0.4, "mae": 0.3}],
    }
