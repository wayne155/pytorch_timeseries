import pytest
from torch_timeseries.experiment import Experiment, register_model
from torch_timeseries.results.schema import RunResult


def _make_fast_exp_class(name="_FastExp"):
    from dataclasses import dataclass
    from torch_timeseries.experiments.forecast import ForecastExp

    @dataclass
    class _FastExp(ForecastExp):
        dataset_type: str = "ETTh1"
        model_type: str = "DLinear"
        epochs: int = 1
        batch_size: int = 4

        def run(self, seed=1):
            return {"mse": 0.5, "mae": 0.4}

        def _build_model(self):
            import torch.nn as nn
            return nn.Linear(1, 1)

    _FastExp.__name__ = name
    _FastExp.__qualname__ = name
    return _FastExp


def test_experiment_run_returns_run_result(monkeypatch):
    fast_cls = _make_fast_exp_class("_FastModel")
    monkeypatch.setitem(
        __import__("torch_timeseries.experiments", fromlist=["EXPERIMENT_REGISTRY"]).EXPERIMENT_REGISTRY,
        ("_FastModel", "Forecast"),
        fast_cls,
    )
    result = Experiment(model="_FastModel", task="Forecast", dataset="ETTh1").run(seeds=[1])
    assert isinstance(result, list)
    assert all(isinstance(r, RunResult) for r in result)
    assert result[0].model == "_FastModel"
    assert result[0].task == "Forecast"
    assert result[0].dataset == "ETTh1"
    assert result[0].seed == 1
    assert "mse" in result[0].metrics


def test_experiment_run_captures_timestamp(monkeypatch):
    fast_cls = _make_fast_exp_class("_FastModel2")
    monkeypatch.setitem(
        __import__("torch_timeseries.experiments", fromlist=["EXPERIMENT_REGISTRY"]).EXPERIMENT_REGISTRY,
        ("_FastModel2", "Forecast"),
        fast_cls,
    )
    result = Experiment(model="_FastModel2", task="Forecast", dataset="ETTh1").run(seeds=[1])
    assert result[0].timestamp != ""


def test_experiment_with_local_saves_json(tmp_path, monkeypatch):
    fast_cls = _make_fast_exp_class("_FastModel3")
    monkeypatch.setitem(
        __import__("torch_timeseries.experiments", fromlist=["EXPERIMENT_REGISTRY"]).EXPERIMENT_REGISTRY,
        ("_FastModel3", "Forecast"),
        fast_cls,
    )
    Experiment(model="_FastModel3", task="Forecast", dataset="ETTh1") \
        .with_local(save_dir=str(tmp_path)) \
        .run(seeds=[1])
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1


def test_experiment_set_overrides_setting(monkeypatch):
    fast_cls = _make_fast_exp_class("_FastModel4")
    monkeypatch.setitem(
        __import__("torch_timeseries.experiments", fromlist=["EXPERIMENT_REGISTRY"]).EXPERIMENT_REGISTRY,
        ("_FastModel4", "Forecast"),
        fast_cls,
    )
    results = Experiment(model="_FastModel4", task="Forecast", dataset="ETTh1") \
        .set(epochs=2) \
        .run(seeds=[1])
    assert results[0].hparams.get("epochs") == 2


def test_register_model_makes_it_runnable():
    from dataclasses import dataclass
    from torch import nn
    from torch_timeseries.experiments.forecast import ForecastExp

    @dataclass
    class _MyNewModel(ForecastExp):
        dataset_type: str = "ETTh1"
        model_type: str = "_MyNewModel"
        epochs: int = 1

        def run(self, seed=1):
            return {"mse": 0.42}

        def _build_model(self):
            return nn.Linear(1, 1)

    register_model(_MyNewModel)

    results = Experiment(model="_MyNewModel", task="Forecast", dataset="ETTh1").run(seeds=[1])
    assert len(results) == 1
    assert results[0].model == "_MyNewModel"


def test_experiment_grid_runs_all_combos(monkeypatch):
    from dataclasses import dataclass
    from torch import nn
    from torch_timeseries.experiments.forecast import ForecastExp
    from torch_timeseries.experiments import EXPERIMENT_REGISTRY

    @dataclass
    class _GridModel(ForecastExp):
        dataset_type: str = "ETTh1"
        model_type: str = "_GridModel"
        epochs: int = 1

        def run(self, seed=1):
            return {"mse": 0.5}

        def _build_model(self):
            return nn.Linear(1, 1)

    EXPERIMENT_REGISTRY[("_GridModel", "Forecast")] = _GridModel

    results = Experiment.grid(
        models=["_GridModel"],
        tasks=["Forecast"],
        datasets=["ETTh1", "ETTm1"],
        seeds=[1, 2],
    ).run()

    assert len(results) == 4    # 1 model × 1 task × 2 datasets × 2 seeds


def test_experiment_compare_prints_table(tmp_path, capsys):
    from torch_timeseries.results.backends import LocalBackend
    from torch_timeseries.results.schema import RunResult

    backend = LocalBackend(save_dir=str(tmp_path))
    for seed in [1, 2]:
        backend.save(RunResult(
            model="DLinear", task="Forecast", dataset="ETTh1", seed=seed,
            timestamp="2026-01-01T00:00:00",
            hparams={}, metrics={"mse": 0.38 + seed * 0.01, "mae": 0.27},
            num_params=22000, train_time_sec=10.0, git_commit="abc",
        ))

    Experiment.compare(save_dir=str(tmp_path), task="Forecast")
    captured = capsys.readouterr().out
    assert "DLinear" in captured
    assert "mse" in captured
    assert "ETTh1" in captured
