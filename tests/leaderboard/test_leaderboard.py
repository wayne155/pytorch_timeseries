import csv
import json
import subprocess
import sys

import pytest
import yaml


def _write_run(path, *, model, task, dataset, seed, metrics, hparams=None):
    path.write_text(
        json.dumps(
            {
                "model": model,
                "task": task,
                "dataset": dataset,
                "seed": seed,
                "timestamp": "2026-01-01T00:00:00",
                "hparams": hparams or {"windows": 96, "pred_len": 96},
                "metrics": metrics,
                "num_params": 1234,
                "train_time_sec": 12.5,
                "git_commit": "abc123",
                "history": None,
            }
        )
    )


def test_load_local_run_results_as_leaderboard_entries(tmp_path):
    from torch_timeseries.leaderboard.loaders import load_local_entries

    _write_run(
        tmp_path / "DLinear_Forecast_ETTh1_seed1.json",
        model="DLinear",
        task="Forecast",
        dataset="ETTh1",
        seed=1,
        metrics={"mse": 0.4, "mae": 0.3},
    )

    entries = load_local_entries(str(tmp_path))

    assert len(entries) == 1
    entry = entries[0]
    assert entry.model == "DLinear"
    assert entry.task == "Forecast"
    assert entry.dataset == "ETTh1"
    assert entry.source.source_type == "local"
    assert entry.source.source_name == "RunResult"
    assert entry.metrics["mse"] == pytest.approx(0.4)
    assert entry.seed == 1


def test_load_curated_yaml_entries_with_source_metadata(tmp_path):
    from torch_timeseries.leaderboard.loaders import load_curated_entries

    entries_dir = tmp_path / "entries"
    entries_dir.mkdir()
    (entries_dir / "forecast.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "model": "PatchTST",
                    "task": "Forecast",
                    "dataset": "ETTh1",
                    "hparams": {"windows": 96, "pred_len": 96},
                    "metrics": {"mse": {"mean": 0.37, "std": 0.01}, "mae": 0.25},
                    "num_seeds": 3,
                    "source": {
                        "source_type": "external",
                        "source_name": "PatchTST paper",
                        "citation": "Nie et al. 2023",
                        "url": "https://example.com/patchtst",
                        "notes": "Reported table value.",
                    },
                }
            ]
        )
    )

    entries = load_curated_entries(str(entries_dir))

    assert len(entries) == 1
    entry = entries[0]
    assert entry.source.source_type == "external"
    assert entry.source.citation == "Nie et al. 2023"
    assert entry.metric_mean["mse"] == pytest.approx(0.37)
    assert entry.metric_std["mse"] == pytest.approx(0.01)
    assert entry.num_seeds == 3


def test_aggregate_and_rank_task_specific_metrics():
    from torch_timeseries.leaderboard.ranking import aggregate_entries, rank_entries
    from torch_timeseries.leaderboard.schema import LeaderboardEntry, LeaderboardSource

    source = LeaderboardSource(source_type="local", source_name="RunResult")
    entries = [
        LeaderboardEntry(
            model="A",
            task="Forecast",
            dataset="ETTh1",
            hparams={"pred_len": 96},
            metrics={"mse": 0.4, "mae": 0.3},
            source=source,
            seed=1,
        ),
        LeaderboardEntry(
            model="A",
            task="Forecast",
            dataset="ETTh1",
            hparams={"pred_len": 96},
            metrics={"mse": 0.6, "mae": 0.5},
            source=source,
            seed=2,
        ),
        LeaderboardEntry(
            model="B",
            task="Forecast",
            dataset="ETTh1",
            hparams={"pred_len": 96},
            metrics={"mse": 0.3, "mae": 0.2},
            source=source,
            seed=1,
        ),
        LeaderboardEntry(
            model="C",
            task="UEAClassification",
            dataset="EthanolConcentration",
            hparams={},
            metrics={"accuracy": 0.8},
            source=source,
            seed=1,
        ),
        LeaderboardEntry(
            model="D",
            task="UEAClassification",
            dataset="EthanolConcentration",
            hparams={},
            metrics={"accuracy": 0.9},
            source=source,
            seed=1,
        ),
    ]

    aggregated = aggregate_entries(entries)
    forecast = rank_entries([e for e in aggregated if e.task == "Forecast"])
    classification = rank_entries([e for e in aggregated if e.task == "UEAClassification"])

    assert forecast[0].model == "B"
    assert forecast[0].rank == 1
    assert forecast[1].model == "A"
    assert forecast[1].metric_mean["mse"] == pytest.approx(0.5)
    assert forecast[1].metric_std["mse"] == pytest.approx(0.1414213562)
    assert forecast[1].num_seeds == 2

    assert classification[0].model == "D"
    assert classification[0].rank == 1


def test_render_markdown_csv_json_deterministically(tmp_path):
    from torch_timeseries.leaderboard.render import write_leaderboard_outputs
    from torch_timeseries.leaderboard.schema import LeaderboardEntry, LeaderboardSource

    entries = [
        LeaderboardEntry(
            model="DLinear",
            task="Forecast",
            dataset="ETTh1",
            hparams={"pred_len": 96},
            metrics={"mse": 0.4, "mae": 0.3},
            metric_mean={"mse": 0.4, "mae": 0.3},
            metric_std={"mse": 0.0, "mae": 0.0},
            num_seeds=1,
            source=LeaderboardSource(source_type="local", source_name="RunResult"),
            rank=1,
        ),
        LeaderboardEntry(
            model="DLinear",
            task="Imputation",
            dataset="ETTh1",
            hparams={"windows": 96, "mask_ratio": 0.25},
            metrics={"mse": 0.2},
            metric_mean={"mse": 0.2},
            metric_std={"mse": 0.0},
            num_seeds=1,
            source=LeaderboardSource(source_type="local", source_name="RunResult"),
            rank=1,
        ),
    ]

    write_leaderboard_outputs(
        entries,
        output_dir=str(tmp_path / "out"),
        docs_dir=str(tmp_path / "docs"),
    )

    csv_path = tmp_path / "out" / "leaderboard.csv"
    json_path = tmp_path / "out" / "leaderboard.json"
    docs_index = tmp_path / "docs" / "index.md"

    assert csv_path.exists()
    assert json_path.exists()
    assert docs_index.exists()
    assert "## Forecast" in docs_index.read_text()
    assert "## Imputation" in docs_index.read_text()

    rows = list(csv.DictReader(csv_path.open()))
    assert rows[0]["task"] == "Forecast"
    assert rows[0]["primary_metric"] == "mse"
    assert rows[0]["metric_mean"] == "0.4"

    data = json.loads(json_path.read_text())
    assert data[0]["model"] == "DLinear"
    assert data[0]["rank"] == 1


def test_cli_generates_leaderboard_outputs(tmp_path):
    results_dir = tmp_path / "results"
    entries_dir = tmp_path / "entries"
    output_dir = tmp_path / "leaderboard"
    docs_dir = tmp_path / "docs"
    results_dir.mkdir()
    entries_dir.mkdir()

    _write_run(
        results_dir / "DLinear_Forecast_ETTh1_seed1.json",
        model="DLinear",
        task="Forecast",
        dataset="ETTh1",
        seed=1,
        metrics={"mse": 0.4, "mae": 0.3},
    )
    (entries_dir / "classification.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "model": "ExternalClassifier",
                    "task": "UEAClassification",
                    "dataset": "EthanolConcentration",
                    "metrics": {"accuracy": 0.9},
                    "source": {"source_type": "external", "source_name": "paper"},
                }
            ]
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "./torch_timeseries/cli/exp.py",
            "leaderboard",
            "--results_dir",
            str(results_dir),
            "--entries_dir",
            str(entries_dir),
            "--output_dir",
            str(output_dir),
            "--docs_dir",
            str(docs_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "leaderboard.csv").exists()
    assert (output_dir / "leaderboard.json").exists()
    assert (docs_dir / "index.md").exists()
    assert "Wrote leaderboard" in result.stdout
    assert "LeaderboardEntry" not in result.stdout


def test_docs_markdown_contains_all_supported_task_sections(tmp_path):
    from torch_timeseries.leaderboard.render import write_leaderboard_outputs

    write_leaderboard_outputs([], output_dir=str(tmp_path / "out"), docs_dir=str(tmp_path / "docs"))

    text = (tmp_path / "docs" / "index.md").read_text()
    for task in ["Forecast", "Imputation", "AnomalyDetection", "UEAClassification"]:
        assert f"## {task}" in text
