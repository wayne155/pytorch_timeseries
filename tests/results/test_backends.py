import json
import pytest
from torch_timeseries.results.schema import RunResult
from torch_timeseries.results.backends import LocalBackend


def _r(**kw):
    base = dict(
        model="DLinear", task="Forecast", dataset="ETTh1", seed=1,
        timestamp="2026-01-01T00:00:00",
        hparams={"lr": 0.001, "windows": 96},
        metrics={"mse": 0.382, "mae": 0.271},
        num_params=22000, train_time_sec=12.5, git_commit="abc123",
    )
    base.update(kw)
    return RunResult(**base)


def test_run_result_has_required_fields():
    r = _r()
    assert r.history is None
    assert isinstance(r.hparams, dict)
    assert isinstance(r.metrics, dict)


def test_local_backend_saves_json(tmp_path):
    backend = LocalBackend(save_dir=str(tmp_path))
    backend.save(_r())
    fname = tmp_path / "DLinear_Forecast_ETTh1_seed1.json"
    assert fname.exists()
    d = json.loads(fname.read_text())
    assert d["model"] == "DLinear"
    assert d["metrics"]["mse"] == pytest.approx(0.382)
    assert d["history"] is None


def test_local_backend_load_all_no_filter(tmp_path):
    backend = LocalBackend(save_dir=str(tmp_path))
    backend.save(_r(seed=1))
    backend.save(_r(seed=2))
    backend.save(_r(model="Autoformer", seed=1))
    assert len(backend.load_all()) == 3


def test_local_backend_load_all_with_filter(tmp_path):
    backend = LocalBackend(save_dir=str(tmp_path))
    backend.save(_r(seed=1))
    backend.save(_r(seed=2))
    backend.save(_r(model="Autoformer", seed=1))
    results = backend.load_all(model="DLinear")
    assert len(results) == 2
    assert all(r.model == "DLinear" for r in results)


def test_local_backend_overwrite(tmp_path):
    backend = LocalBackend(save_dir=str(tmp_path))
    backend.save(_r(metrics={"mse": 0.5}))
    backend.save(_r(metrics={"mse": 0.3}))
    results = backend.load_all()
    assert len(results) == 1
    assert results[0].metrics["mse"] == pytest.approx(0.3)
