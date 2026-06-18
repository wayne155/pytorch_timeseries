"""Utilities for aggregating and comparing experimental results.

Typical usage::

    from torch_timeseries.results import LocalBackend, ResultsComparator

    backend = LocalBackend("./results")
    cmp = ResultsComparator(backend.load_all(task="Forecast"))
    print(cmp.summary())          # mean ± std across seeds
    df = cmp.to_dataframe()       # pandas DataFrame
    cmp.print_table()             # PrettyTable to stdout
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from prettytable import PrettyTable

from .schema import RunResult


def _mean_std(values: Sequence[float]):
    if not values:
        return float("nan"), float("nan")
    n = len(values)
    mu = sum(values) / n
    if n == 1:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return mu, var ** 0.5


class ResultsComparator:
    """Aggregate a list of :class:`RunResult` objects across seeds.

    Groups results by ``(model, dataset)`` (and optionally ``task``), then
    computes mean and std of each metric across seeds.

    Args:
        results: List of ``RunResult`` objects to aggregate.
        group_by_task: Whether to include ``task`` in the grouping key.
            Default: ``True``.
        metrics: Metric keys to include.  ``None`` includes all metrics
            found in the first result.
    """

    def __init__(
        self,
        results: List[RunResult],
        group_by_task: bool = True,
        metrics: Optional[List[str]] = None,
    ) -> None:
        self.results = results
        self.group_by_task = group_by_task
        self._metrics = metrics
        self._groups: Dict[tuple, List[RunResult]] = defaultdict(list)
        for r in results:
            key = (r.model, r.dataset, r.task) if group_by_task else (r.model, r.dataset)
            self._groups[key].append(r)

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def metric_names(self) -> List[str]:
        if self._metrics is not None:
            return self._metrics
        if not self.results:
            return []
        return sorted(self.results[0].metrics.keys())

    def summary(self) -> Dict[tuple, Dict[str, tuple]]:
        """Return ``{group_key: {metric_name: (mean, std)}}``.

        Example::

            for key, stats in cmp.summary().items():
                model, dataset, task = key
                mse_mean, mse_std = stats["mse"]
        """
        out = {}
        for key, runs in self._groups.items():
            stats = {}
            for m in self.metric_names:
                vals = [r.metrics[m] for r in runs if m in r.metrics]
                stats[m] = _mean_std(vals)
            stats["n_seeds"] = (len(runs), 0)
            out[key] = stats
        return out

    def to_dataframe(self):
        """Return a ``pandas.DataFrame`` with columns:

        ``model``, ``dataset``, (``task``), ``n_seeds``, then one column per
        metric in ``mean / std`` style (e.g. ``mse_mean``, ``mse_std``).

        Raises ``ImportError`` if pandas is not installed.
        """
        import pandas as pd

        rows = []
        for key, stats in self.summary().items():
            row: dict = {}
            if self.group_by_task:
                row["model"], row["dataset"], row["task"] = key
            else:
                row["model"], row["dataset"] = key
            row["n_seeds"] = int(stats["n_seeds"][0])
            for m in self.metric_names:
                mean, std = stats.get(m, (float("nan"), float("nan")))
                row[f"{m}_mean"] = mean
                row[f"{m}_std"] = std
            rows.append(row)
        return pd.DataFrame(rows)

    def print_table(
        self,
        metrics: Optional[List[str]] = None,
        fmt: str = ".4f",
    ) -> None:
        """Print a PrettyTable comparing models.

        Args:
            metrics: Subset of metrics to show.  ``None`` shows all.
            fmt: Python format spec for floats.  Default: ``'.4f'``.
        """
        mets = metrics or self.metric_names
        cols = (
            ["model", "dataset", "task", "seeds"] if self.group_by_task
            else ["model", "dataset", "seeds"]
        )
        for m in mets:
            cols.append(m)
        table = PrettyTable(cols)
        table.align = "l"

        for key, stats in sorted(self.summary().items()):
            row = list(key) + [int(stats["n_seeds"][0])]
            for m in mets:
                mean, std = stats.get(m, (float("nan"), float("nan")))
                row.append(f"{mean:{fmt}} ± {std:{fmt}}")
            table.add_row(row)
        print(table)

    def best_model(
        self,
        metric: str,
        dataset: Optional[str] = None,
        task: Optional[str] = None,
        lower_is_better: bool = True,
    ) -> Optional[str]:
        """Return the model name with the best mean *metric* value.

        Args:
            metric: Metric key to rank by.
            dataset: Filter to a specific dataset.  ``None`` considers all.
            task: Filter to a specific task.  ``None`` considers all.
            lower_is_better: If ``True``, returns the model with the lowest
                mean.  Default: ``True``.

        Returns:
            Model name string, or ``None`` if no results match.
        """
        candidates = {}
        for key, stats in self.summary().items():
            k_model = key[0]
            k_dataset = key[1]
            k_task = key[2] if self.group_by_task else None
            if dataset is not None and k_dataset != dataset:
                continue
            if task is not None and k_task != task:
                continue
            if metric not in stats:
                continue
            mean, _ = stats[metric]
            if k_model not in candidates:
                candidates[k_model] = []
            candidates[k_model].append(mean)
        if not candidates:
            return None
        avg_by_model = {m: sum(vs) / len(vs) for m, vs in candidates.items()}
        return min(avg_by_model, key=avg_by_model.__getitem__) if lower_is_better \
            else max(avg_by_model, key=avg_by_model.__getitem__)
