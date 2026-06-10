from __future__ import annotations

import importlib.util
from pathlib import Path


REPRODUCE_ROOT = Path("leaderboard/reproduce")

TASK_DIRS = {
    "anomaly_detection": "AnomalyDetection",
    "imputation": "Imputation",
    "long_term_forecast": "Forecast",
    "short_term_forecast": "Forecast",
    "uea_classification": "UEAClassification",
}

MODELS = {
    "autoformer": "Autoformer",
    "dlinear": "DLinear",
    "fedformer": "FEDformer",
    "itransformer": "iTransformer",
    "nlinear": "NLinear",
    "patchtst": "PatchTST",
    "timesnet": "TimesNet",
}


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location(
        f"reproduce_{path.parent.name}_{path.stem}",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_requested_reproduce_script_matrix_exists():
    missing = [
        str(REPRODUCE_ROOT / task_dir / f"{model_file}.py")
        for task_dir in TASK_DIRS
        for model_file in MODELS
        if not (REPRODUCE_ROOT / task_dir / f"{model_file}.py").exists()
    ]

    assert missing == []


def test_reproduce_scripts_expose_consistent_metadata():
    for task_dir, expected_task in TASK_DIRS.items():
        for model_file, expected_model in MODELS.items():
            module = _load_script(REPRODUCE_ROOT / task_dir / f"{model_file}.py")

            assert module.MODEL == expected_model
            assert module.TASK == expected_task
            assert isinstance(module.DATASETS, list)
            assert module.DATASETS
            assert isinstance(module.SEEDS, list)
            assert module.SEEDS
            assert isinstance(module.PARAMS, dict)
            assert hasattr(module, "GRID")


def test_nlinear_is_registered_for_supported_requested_tasks():
    from torch_timeseries.experiments import list_experiments

    available = set(list_experiments())

    assert ("NLinear", "Forecast") in available
    assert ("NLinear", "Imputation") in available
    assert ("NLinear", "AnomalyDetection") in available
    assert ("NLinear", "UEAClassification") in available


def test_run_matrix_supports_non_pred_len_axes(monkeypatch):
    from leaderboard.reproduce import _runner

    captured = {}

    def fake_run_jobs(model, task, jobs, params, device):
        captured["model"] = model
        captured["task"] = task
        captured["jobs"] = jobs
        captured["params"] = params
        captured["device"] = device

    monkeypatch.setattr(_runner, "_run_jobs", fake_run_jobs)
    monkeypatch.setattr(_runner.subprocess, "run", lambda *args, **kwargs: None)

    _runner.run_matrix(
        model="DLinear",
        task="Imputation",
        datasets=["ETTh1"],
        seeds=[0],
        params={"windows": 96},
        grid={"mask_rate": [0.125, 0.5]},
        argv=[
            "--cpu",
            "--datasets",
            "ETTh2",
            "--seeds",
            "3",
            "--mask-rates",
            "0.25",
            "0.375",
        ],
        build=False,
    )

    assert captured == {
        "model": "DLinear",
        "task": "Imputation",
        "jobs": [
            ("ETTh2", {"mask_rate": 0.25}, 3),
            ("ETTh2", {"mask_rate": 0.375}, 3),
        ],
        "params": {"windows": 96},
        "device": "cpu",
    }
