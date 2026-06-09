# tests/leaderboard/test_build_task_view.py
from __future__ import annotations
import json
import pathlib
import textwrap
import pytest
from scripts.build_leaderboard import (
    collect_results,
    matches_hparams,
    aggregate_metrics,
    mean_of_agg,
    build,
    load_config,
)


@pytest.fixture()
def lr_tree(tmp_path):
    """Create a leaderboard_results/ tree with 3 seeds for DLinear/Forecast/ETTh1."""
    for seed in range(1, 4):
        p = tmp_path / "DLinear" / "Forecast" / "ETTh1" / f"seed{seed}" / "metrics.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "model": "DLinear", "task": "Forecast", "dataset": "ETTh1",
            "seed": seed, "hparams": {"windows": 96, "pred_len": 96},
            "metrics": {"mse": 0.38 + seed * 0.01, "mae": 0.28 + seed * 0.005},
        }))
    return tmp_path


@pytest.fixture()
def view_yaml(tmp_path):
    content = textwrap.dedent("""\
        id: long_term_forecast
        display_name: "Long-Term Forecast"
        task: Forecast
        primary_metrics: [mse, mae]
        variants:
          - label: "I96"
            match: {windows: 96}
            subcolumns:
              - {label: "96", match: {pred_len: 96}}
              - {label: "192", match: {pred_len: 192}}
        datasets:
          - ETTh1
    """)
    d = tmp_path / "views"
    d.mkdir()
    (d / "long_term_forecast.yaml").write_text(content)
    return d


def test_collect_results_from_tree(lr_tree, tmp_path):
    results = collect_results(lr_tree, tmp_path / "empty_results", "Forecast", "ETTh1")
    assert len(results) == 3
    assert all(r["model"] == "DLinear" for r in results)


def test_collect_results_deduplication(lr_tree, tmp_path):
    """leaderboard_results takes priority over results/."""
    flat = tmp_path / "results"
    flat.mkdir()
    (flat / "extra.json").write_text(json.dumps({
        "model": "DLinear", "task": "Forecast", "dataset": "ETTh1",
        "seed": 1, "hparams": {"windows": 96, "pred_len": 96},
        "metrics": {"mse": 0.99, "mae": 0.99},
    }))
    results = collect_results(lr_tree, flat, "Forecast", "ETTh1")
    seed1 = next(r for r in results if r["seed"] == 1)
    assert seed1["metrics"]["mse"] != 0.99  # tree version wins


def test_matches_hparams():
    data = {"hparams": {"windows": 96, "pred_len": 96}}
    assert matches_hparams(data, {"windows": 96}) is True
    assert matches_hparams(data, {"windows": 96, "pred_len": 96}) is True
    assert matches_hparams(data, {"windows": 96, "pred_len": 192}) is False
    assert matches_hparams(data, {}) is True


def test_aggregate_metrics_mean_std():
    seeds = [
        {"metrics": {"mse": 0.38, "mae": 0.28}},
        {"metrics": {"mse": 0.40, "mae": 0.30}},
        {"metrics": {"mse": 0.42, "mae": 0.32}},
    ]
    agg = aggregate_metrics(seeds)
    assert abs(agg["mse"]["mean"] - 0.40) < 1e-9
    assert agg["mse"]["n_seeds"] == 3
    assert agg["mse"]["std"] > 0


def test_mean_of_agg():
    subcol_aggs = [
        {"mse": {"mean": 0.38, "std": 0.01, "n_seeds": 3}},
        {"mse": {"mean": 0.42, "std": 0.02, "n_seeds": 3}},
    ]
    avg = mean_of_agg(subcol_aggs)
    assert abs(avg["mse"]["mean"] - 0.40) < 1e-9


def test_paper_entry_used_when_no_local_runs(view_yaml, tmp_path):
    """Paper entries are used when no local-run data exists for that model."""
    paper_entries = [{
        "model": "DLinear",
        "task": "Forecast",
        "dataset": "ETTh1",
        "seed": None,
        "hparams": {"windows": 96, "pred_len": 96},
        "metrics": {"mse": 0.40, "mae": 0.30},
        "source_type": "paper",
        "citation": "DLinear (Zeng 2023)",
        "url": "",
    }]
    from scripts.build_leaderboard import _ingest_entry_yaml, _build_view
    import pathlib
    view_cfg_path = list(pathlib.Path(str(view_yaml)).glob("*.yaml"))[0]
    import yaml
    view_cfg = yaml.safe_load(view_cfg_path.read_text())
    view_block = _build_view(
        view_cfg,
        tmp_path / "empty_lr",
        tmp_path / "empty_results",
        paper_entries,
    )
    assert len(view_block["models"]) == 1
    model = view_block["models"][0]
    assert model["name"] == "DLinear"
    assert model["source_type"] == "paper"
    eth1 = model["results"]["I96"]["ETTh1"]
    assert "96" in eth1  # subcolumn 96 should have DLinear's pred_len=96 result
    assert eth1["96"]["mse"]["n_seeds"] == 1
    assert abs(eth1["96"]["mse"]["mean"] - 0.40) < 1e-9


def test_build_end_to_end(lr_tree, view_yaml, tmp_path):
    cfg = {
        "leaderboard_results_dir": str(lr_tree),
        "results_dir": str(tmp_path / "empty"),
        "views_dir": str(view_yaml),
        "entries_dir": str(tmp_path / "empty_entries"),
        "out": str(tmp_path / "out.json"),
    }
    build(cfg)
    data = json.loads((tmp_path / "out.json").read_text())
    assert "views" in data
    assert len(data["views"]) == 1
    view = data["views"][0]
    assert view["id"] == "long_term_forecast"
    assert len(view["models"]) == 1
    model = view["models"][0]
    assert model["name"] == "DLinear"
    eth1 = model["results"]["I96"]["ETTh1"]
    assert "avg" in eth1
    assert "96" in eth1
    assert eth1["96"]["mse"]["n_seeds"] == 3
