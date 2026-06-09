from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

from .ranking import primary_metric_for_task
from .schema import SUPPORTED_TASKS, LeaderboardEntry


CSV_FIELDS = [
    "rank",
    "task",
    "dataset",
    "model",
    "primary_metric",
    "metric_mean",
    "metric_std",
    "num_seeds",
    "source_type",
    "source_name",
    "hparams",
    "metrics",
    "citation",
    "url",
    "notes",
    "num_params",
    "train_time_sec",
    "git_commit",
]


def _primary_values(entry: LeaderboardEntry):
    metric, _direction = primary_metric_for_task(entry.task, entry.metric_mean)
    return metric, entry.metric_mean.get(metric), entry.metric_std.get(metric)


def _row(entry: LeaderboardEntry) -> dict:
    primary_metric, metric_mean, metric_std = _primary_values(entry)
    return {
        "rank": entry.rank or "",
        "task": entry.task,
        "dataset": entry.dataset,
        "model": entry.model,
        "primary_metric": primary_metric,
        "metric_mean": "" if metric_mean is None else str(metric_mean),
        "metric_std": "" if metric_std is None else str(metric_std),
        "num_seeds": entry.num_seeds,
        "source_type": entry.source.source_type,
        "source_name": entry.source.source_name,
        "hparams": json.dumps(entry.hparams, sort_keys=True),
        "metrics": json.dumps(entry.metric_mean, sort_keys=True),
        "citation": entry.source.citation,
        "url": entry.source.url,
        "notes": entry.source.notes,
        "num_params": "" if entry.num_params is None else entry.num_params,
        "train_time_sec": "" if entry.train_time_sec is None else entry.train_time_sec,
        "git_commit": entry.git_commit,
    }


def _format_metric(mean, std):
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.6g}"
    return f"{mean:.6g} ± {std:.3g}"


def render_markdown(entries) -> str:
    by_task = defaultdict(list)
    for entry in entries:
        by_task[entry.task].append(entry)

    lines = [
        "# Time-Series Leaderboard",
        "",
        "Static leaderboard generated from local RunResult JSON files and curated YAML entries.",
        "",
    ]
    for task in SUPPORTED_TASKS:
        lines.extend([f"## {task}", ""])
        task_entries = by_task.get(task, [])
        if not task_entries:
            lines.extend(["No entries yet.", ""])
            continue
        lines.extend(
            [
                "| Rank | Dataset | Model | Metric | Seeds | Source |",
                "| ---: | --- | --- | --- | ---: | --- |",
            ]
        )
        for entry in task_entries:
            primary_metric, mean, std = _primary_values(entry)
            metric_text = f"{primary_metric}={_format_metric(mean, std)}"
            source = f"{entry.source.source_type}:{entry.source.source_name}"
            lines.append(
                f"| {entry.rank or ''} | {entry.dataset} | {entry.model} | "
                f"{metric_text} | {entry.num_seeds} | {source} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_leaderboard_outputs(
    entries,
    output_dir: str = "results/leaderboard",
    docs_dir: str = "docs/leaderboard",
) -> None:
    entries = sorted(
        list(entries),
        key=lambda e: (e.task, e.dataset, e.rank or 999999, e.model, e.source.source_name),
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)

    rows = [_row(entry) for entry in entries]
    with open(os.path.join(output_dir, "leaderboard.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(output_dir, "leaderboard.json"), "w") as f:
        json.dump([entry.as_dict() for entry in entries], f, indent=2)

    markdown = render_markdown(entries)
    with open(os.path.join(docs_dir, "index.md"), "w") as f:
        f.write(markdown)
    with open(os.path.join(docs_dir, "index.rst"), "w") as f:
        f.write(
            "Time-Series Leaderboard\n"
            "=======================\n\n"
            ".. raw:: html\n\n"
            "   <p>See <code>index.md</code> for the generated Markdown leaderboard.</p>\n"
        )
