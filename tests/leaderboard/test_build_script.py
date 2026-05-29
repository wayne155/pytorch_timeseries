import json
import pathlib
import textwrap
import pytest
from scripts.build_leaderboard import (
    ingest_result_json,
    ingest_yaml_entries,
    build_schema,
    build,
    make_id,
)


@pytest.fixture()
def result_json(tmp_path):
    data = {
        "model": "PatchTST", "task": "Forecast", "dataset": "ETTh1", "seed": 1,
        "timestamp": "2026-01-01T00:00:00", "git_commit": "abc123",
        "hparams": {
            "windows": 96, "pred_len": 96, "horizon": 1,
            "device": "cpu", "batch_size": 32, "lr": 1e-4,
        },
        "metrics": {"mse": 0.38, "mae": 0.28},
        "num_params": 100000, "train_time_sec": 60.0, "history": None,
    }
    p = tmp_path / "PatchTST_Forecast_ETTh1_seed1.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture()
def yaml_entry(tmp_path):
    content = textwrap.dedent("""\
        - model: DLinear
          task: Forecast
          dataset: ETTh1
          hparams: {windows: 96, pred_len: 96, horizon: 1}
          metrics: {mse: 0.40, mae: 0.30}
          source_type: paper
          citation: "DLinear (Zeng 2023)"
          url: https://arxiv.org/abs/2205.13504
          notes: "verified"
    """)
    p = tmp_path / "dlinear.yaml"
    p.write_text(content)
    return p


def test_ingest_result_json_fields(result_json):
    e = ingest_result_json(result_json)
    assert e["model"] == "PatchTST"
    assert e["task"] == "Forecast"
    assert e["dataset"] == "ETTh1"
    assert e["seed"] == 1
    assert e["source_type"] == "local_run"
    assert e["metrics"] == {"mse": 0.38, "mae": 0.28}
    assert e["num_params"] == 100000


def test_ingest_result_json_strips_infra_hparams(result_json):
    e = ingest_result_json(result_json)
    assert set(e["hparams"].keys()) == {"windows", "pred_len", "horizon"}
    assert "device" not in e["hparams"]
    assert "batch_size" not in e["hparams"]


def test_ingest_yaml_entry_fields(yaml_entry):
    entries = ingest_yaml_entries(yaml_entry)
    assert len(entries) == 1
    e = entries[0]
    assert e["model"] == "DLinear"
    assert e["seed"] is None
    assert e["source_type"] == "paper"
    assert e["citation"] == "DLinear (Zeng 2023)"
    assert e["url"] == "https://arxiv.org/abs/2205.13504"


def test_schema_hparam_options(result_json, yaml_entry):
    entries = [ingest_result_json(result_json)] + ingest_yaml_entries(yaml_entry)
    schema = build_schema(entries)
    assert "Forecast" in schema["tasks"]
    assert "ETTh1" in schema["datasets_by_task"]["Forecast"]
    assert "windows" in schema["hparams_by_task"]["Forecast"]
    assert 96 in schema["hparam_options"]["Forecast"]["windows"]
    assert "PatchTST" in schema["models"]
    assert "DLinear" in schema["models"]


def test_build_writes_one_entry_per_seed(result_json, yaml_entry, tmp_path):
    out = tmp_path / "leaderboard_data.json"
    build(result_json.parent, yaml_entry.parent, out)
    data = json.loads(out.read_text())
    # One local_run + one paper = 2 entries (no aggregation)
    assert len(data["entries"]) == 2
    assert any(e["seed"] == 1 for e in data["entries"])
    assert any(e["seed"] is None for e in data["entries"])


def test_build_idempotent(result_json, yaml_entry, tmp_path):
    out = tmp_path / "leaderboard_data.json"
    build(result_json.parent, yaml_entry.parent, out)
    ids_first = [e["id"] for e in json.loads(out.read_text())["entries"]]
    build(result_json.parent, yaml_entry.parent, out)
    ids_second = [e["id"] for e in json.loads(out.read_text())["entries"]]
    assert ids_first == ids_second


def test_make_id_deterministic():
    a = make_id("M", "T", "D", 1, {"k": 1}, "local_run")
    b = make_id("M", "T", "D", 1, {"k": 1}, "local_run")
    assert a == b
    assert len(a) == 16
