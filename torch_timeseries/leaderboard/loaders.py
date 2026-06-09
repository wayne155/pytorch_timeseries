from __future__ import annotations

import os
from typing import List

import yaml

from torch_timeseries.results.backends import LocalBackend

from .schema import LeaderboardEntry, LeaderboardSource


def load_local_entries(results_dir: str = "./results") -> List[LeaderboardEntry]:
    if not os.path.isdir(results_dir):
        return []
    backend = LocalBackend(save_dir=results_dir)
    entries = []
    for result in backend.load_all():
        entries.append(
            LeaderboardEntry(
                model=result.model,
                task=result.task,
                dataset=result.dataset,
                hparams=result.hparams,
                metrics=result.metrics,
                source=LeaderboardSource(
                    source_type="local",
                    source_name="RunResult",
                    notes=f"git_commit={result.git_commit}",
                ),
                seed=result.seed,
                num_params=result.num_params,
                train_time_sec=result.train_time_sec,
                git_commit=result.git_commit,
            )
        )
    return entries


def _metric_parts(metrics: dict):
    flat_metrics = {}
    metric_mean = {}
    metric_std = {}
    for name, value in (metrics or {}).items():
        if isinstance(value, dict):
            mean_value = value.get("mean")
            std_value = value.get("std", 0.0)
            flat_metrics[name] = mean_value
            metric_mean[name] = mean_value
            metric_std[name] = std_value
        else:
            flat_metrics[name] = value
            metric_mean[name] = value
            metric_std[name] = 0.0
    return flat_metrics, metric_mean, metric_std


def load_curated_entries(entries_dir: str = "leaderboard/entries") -> List[LeaderboardEntry]:
    if not os.path.isdir(entries_dir):
        return []

    entries = []
    for fname in sorted(os.listdir(entries_dir)):
        if not fname.endswith((".yaml", ".yml")):
            continue
        path = os.path.join(entries_dir, fname)
        with open(path) as f:
            data = yaml.safe_load(f) or []
        if isinstance(data, dict):
            data = data.get("entries", [])
        for item in data:
            source_data = item.get("source", {})
            metrics, metric_mean, metric_std = _metric_parts(item.get("metrics", {}))
            entries.append(
                LeaderboardEntry(
                    model=item["model"],
                    task=item["task"],
                    dataset=item["dataset"],
                    hparams=item.get("hparams", {}),
                    metrics=metrics,
                    metric_mean=metric_mean,
                    metric_std=metric_std,
                    num_seeds=int(item.get("num_seeds", 1)),
                    seed=item.get("seed"),
                    source=LeaderboardSource(
                        source_type=source_data.get("source_type", "external"),
                        source_name=source_data.get("source_name", fname),
                        citation=source_data.get("citation", ""),
                        url=source_data.get("url", ""),
                        notes=source_data.get("notes", ""),
                    ),
                    num_params=item.get("num_params"),
                    train_time_sec=item.get("train_time_sec"),
                    git_commit=item.get("git_commit", ""),
                )
            )
    return entries


def load_all_entries(
    results_dir: str = "./results",
    entries_dir: str = "leaderboard/entries",
) -> List[LeaderboardEntry]:
    return load_local_entries(results_dir) + load_curated_entries(entries_dir)
