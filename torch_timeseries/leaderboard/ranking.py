from __future__ import annotations

import json
import math
from collections import defaultdict
from statistics import mean, stdev
from typing import Iterable, List, Tuple

from .schema import LeaderboardEntry, LeaderboardSource


PRIMARY_METRICS = {
    "Forecast": ("mse", "lower"),
    "Imputation": ("mse", "lower"),
    "AnomalyDetection": ("F-score", "higher"),
    "UEAClassification": ("accuracy", "higher"),
}

CLASSIFICATION_ALIASES = ("accuracy", "Accuracy", "MulticlassAccuracy")


def primary_metric_for_task(task: str, metrics=None) -> Tuple[str, str]:
    if task == "UEAClassification" and metrics:
        for name in CLASSIFICATION_ALIASES:
            if name in metrics:
                return name, "higher"
    return PRIMARY_METRICS.get(task, ("mse", "lower"))


def _stable_hparams(hparams: dict) -> str:
    return json.dumps(hparams or {}, sort_keys=True, separators=(",", ":"))


def _group_key(entry: LeaderboardEntry):
    return (
        entry.model,
        entry.task,
        entry.dataset,
        _stable_hparams(entry.hparams),
        entry.source.source_type,
        entry.source.source_name,
    )


def aggregate_entries(entries: Iterable[LeaderboardEntry]) -> List[LeaderboardEntry]:
    groups = defaultdict(list)
    for entry in entries:
        groups[_group_key(entry)].append(entry)

    aggregated = []
    for group in groups.values():
        first = group[0]
        metric_names = sorted({name for entry in group for name in entry.metric_mean})
        metric_mean = {}
        metric_std = {}
        for name in metric_names:
            values = [
                entry.metric_mean[name]
                for entry in group
                if name in entry.metric_mean and entry.metric_mean[name] is not None
            ]
            if not values:
                continue
            metric_mean[name] = mean(values)
            if len(values) > 1:
                metric_std[name] = stdev(values)
            else:
                metric_std[name] = group[0].metric_std.get(name, 0.0)

        seeds = sorted({entry.seed for entry in group if entry.seed is not None})
        num_seeds = len(seeds) if seeds else max(entry.num_seeds for entry in group)
        train_times = [
            entry.train_time_sec for entry in group if entry.train_time_sec is not None
        ]

        aggregated.append(
            LeaderboardEntry(
                model=first.model,
                task=first.task,
                dataset=first.dataset,
                hparams=first.hparams,
                metrics=dict(metric_mean),
                metric_mean=metric_mean,
                metric_std=metric_std,
                num_seeds=num_seeds,
                seed=None,
                source=LeaderboardSource(**first.source.__dict__),
                num_params=first.num_params,
                train_time_sec=sum(train_times) if train_times else first.train_time_sec,
                git_commit=first.git_commit,
            )
        )

    return sorted(
        aggregated,
        key=lambda e: (e.task, e.dataset, _stable_hparams(e.hparams), e.model, e.source.source_name),
    )


def rank_entries(entries: Iterable[LeaderboardEntry]) -> List[LeaderboardEntry]:
    ranked = list(entries)

    def sort_key(entry: LeaderboardEntry):
        metric, direction = primary_metric_for_task(entry.task, entry.metric_mean)
        value = entry.metric_mean.get(metric)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            sortable = math.inf
        elif direction == "lower":
            sortable = value
        else:
            sortable = -value
        return (entry.task, entry.dataset, _stable_hparams(entry.hparams), sortable, entry.model)

    ranked.sort(key=sort_key)

    current_group = None
    current_rank = 0
    for entry in ranked:
        group = (entry.task, entry.dataset, _stable_hparams(entry.hparams))
        if group != current_group:
            current_group = group
            current_rank = 1
        else:
            current_rank += 1
        entry.rank = current_rank

    return ranked


def aggregate_and_rank(entries: Iterable[LeaderboardEntry]) -> List[LeaderboardEntry]:
    return rank_entries(aggregate_entries(entries))
