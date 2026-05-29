#!/usr/bin/env python3
"""
Build leaderboard_data.json from results/*.json and leaderboard/entries/*.yaml.

Usage:
    python scripts/build_leaderboard.py [--results-dir results] \
        [--entries-dir leaderboard/entries] \
        [--out webapp/public/leaderboard_data.json]
"""
import argparse
import hashlib
import json
import pathlib
from datetime import datetime, timezone
from typing import Any

import yaml

# Only these hparam keys are shown per task; infra keys (device, lr, batch_size…) are stripped.
KEY_HPARAMS: dict[str, list[str]] = {
    "Forecast": ["windows", "pred_len", "horizon"],
    "UEAClassification": ["windows"],
    "AnomalyDetection": ["windows"],
    "Imputation": ["windows", "mask_rate"],
    "IrregularClassification": ["windows"],
    "Generation": ["seq_len"],
    "ProbForecast": ["seq_len", "pred_len"],
}


def make_id(
    model: str, task: str, dataset: str,
    seed: int | None, hparams: dict, source_type: str,
) -> str:
    key = json.dumps(
        {"model": model, "task": task, "dataset": dataset,
         "seed": seed, "hparams": hparams, "source_type": source_type},
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def ingest_result_json(path: pathlib.Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    task = data["task"]
    keep = KEY_HPARAMS.get(task, [])
    hparams = {k: v for k, v in data.get("hparams", {}).items() if k in keep}
    return {
        "id": make_id(data["model"], task, data["dataset"], data.get("seed"), hparams, "local_run"),
        "model": data["model"],
        "task": task,
        "dataset": data["dataset"],
        "seed": data.get("seed"),
        "hparams": hparams,
        "metrics": data.get("metrics", {}),
        "num_params": data.get("num_params"),
        "train_time_sec": data.get("train_time_sec"),
        "git_commit": data.get("git_commit", ""),
        "timestamp": data.get("timestamp", ""),
        "source_type": "local_run",
        "citation": "",
        "url": "",
        "notes": "",
    }


def ingest_yaml_entries(path: pathlib.Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text()) or []
    entries = []
    for item in raw:
        task = item["task"]
        hparams = item.get("hparams", {})
        entries.append({
            "id": make_id(item["model"], task, item["dataset"], None, hparams, "paper"),
            "model": item["model"],
            "task": task,
            "dataset": item["dataset"],
            "seed": None,
            "hparams": hparams,
            "metrics": item.get("metrics", {}),
            "num_params": item.get("num_params"),
            "train_time_sec": None,
            "git_commit": "",
            "timestamp": "",
            "source_type": "paper",
            "citation": item.get("citation", ""),
            "url": item.get("url", ""),
            "notes": item.get("notes", ""),
        })
    return entries


def build_schema(entries: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = sorted({e["task"] for e in entries})
    datasets_by_task: dict[str, list] = {}
    hparams_by_task: dict[str, list] = {}
    hparam_options: dict[str, dict] = {}

    for task in tasks:
        te = [e for e in entries if e["task"] == task]
        datasets_by_task[task] = sorted({e["dataset"] for e in te})
        all_keys = sorted({k for e in te for k in e["hparams"]})
        hparams_by_task[task] = all_keys
        hparam_options[task] = {
            key: sorted({e["hparams"][key] for e in te if key in e["hparams"]}, key=str)
            for key in all_keys
        }

    return {
        "tasks": tasks,
        "datasets_by_task": datasets_by_task,
        "models": sorted({e["model"] for e in entries}),
        "hparams_by_task": hparams_by_task,
        "hparam_options": hparam_options,
    }


def build(
    results_dir: pathlib.Path,
    entries_dir: pathlib.Path,
    out_path: pathlib.Path,
) -> None:
    entries: list[dict[str, Any]] = []

    for p in sorted(results_dir.glob("*.json")):
        try:
            entries.append(ingest_result_json(p))
        except (KeyError, json.JSONDecodeError) as exc:
            print(f"Warning: skipping {p.name}: {exc}")

    for p in sorted(entries_dir.glob("*.yaml")):
        try:
            entries.extend(ingest_yaml_entries(p))
        except Exception as exc:
            print(f"Warning: skipping {p.name}: {exc}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
        "schema": build_schema(entries),
    }, indent=2))
    print(f"Wrote {len(entries)} entries → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results", type=pathlib.Path)
    parser.add_argument("--entries-dir", default="leaderboard/entries", type=pathlib.Path)
    parser.add_argument("--out", default="webapp/public/leaderboard_data.json", type=pathlib.Path)
    args = parser.parse_args()
    build(args.results_dir, args.entries_dir, args.out)
