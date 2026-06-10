#!/usr/bin/env python3
"""
Build leaderboard_data.json from leaderboard_results/, results/, and view YAMLs.

Usage:
    python scripts/build_leaderboard.py [--config leaderboard.yaml]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from datetime import datetime, timezone
from typing import Any

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_config(config_path: pathlib.Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text())
    return {
        "leaderboard_results_dir": str(ROOT / raw.get("leaderboard_results_dir", "leaderboard_results")),
        "results_dir": str(ROOT / raw.get("results_dir", "results")),
        "views_dir": str(ROOT / raw.get("views_dir", "leaderboard/views")),
        "entries_dir": str(ROOT / raw.get("entries_dir", "leaderboard/entries")),
        "out": str(ROOT / raw.get("out", "leaderboard/webapp/public/leaderboard_data.json")),
    }


def collect_results(
    leaderboard_results_dir: pathlib.Path,
    results_dir: pathlib.Path,
    task: str,
    dataset: str,
) -> list[dict[str, Any]]:
    """Collect per-seed metric dicts for (task, dataset). leaderboard_results takes priority."""
    seen: dict[tuple, dict] = {}

    def dedup_key(data: dict) -> tuple:
        # hparams must be part of the key: the same (model, seed) legitimately
        # appears once per config (e.g. per pred_len).
        return (
            data.get("model"),
            data.get("seed"),
            json.dumps(data.get("hparams", {}), sort_keys=True),
        )

    # Priority 1: leaderboard_results/{Model}/{Task}/{Dataset}/seed{N}/metrics.json
    if leaderboard_results_dir.exists():
        for model_dir in sorted(leaderboard_results_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            data_dir = model_dir / task / dataset
            if not data_dir.exists():
                continue
            for seed_dir in sorted(data_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                mf = seed_dir / "metrics.json"
                if mf.exists():
                    try:
                        data = json.loads(mf.read_text())
                        seen[dedup_key(data)] = data
                    except (json.JSONDecodeError, KeyError):
                        pass

    # Priority 2: results/*.json flat format (fallback)
    if results_dir.exists():
        for p in sorted(results_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text())
                if data.get("task") == task and data.get("dataset") == dataset:
                    key = dedup_key(data)
                    if key not in seen:
                        seen[key] = data
            except (json.JSONDecodeError, KeyError):
                pass

    return list(seen.values())


def matches_hparams(data: dict[str, Any], match: dict[str, Any]) -> bool:
    """True if data['hparams'] contains all key-value pairs in match."""
    hp = data.get("hparams", {})
    return all(hp.get(k) == v for k, v in match.items())


def aggregate_metrics(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate list of per-seed dicts → {metric: {mean, std, n_seeds}}."""
    if not seed_results:
        return {}
    keys: set[str] = set()
    for r in seed_results:
        keys.update(r.get("metrics", {}).keys())
    out: dict[str, Any] = {}
    for k in sorted(keys):
        vals = [r["metrics"][k] for r in seed_results if k in r.get("metrics", {})]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
        out[k] = {"mean": mean, "std": var ** 0.5, "n_seeds": len(vals)}
    return out


def mean_of_agg(agg_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean of multiple aggregated metric dicts (for avg subcolumn)."""
    if not agg_list:
        return {}
    keys: set[str] = set()
    for a in agg_list:
        keys.update(a.keys())
    out: dict[str, Any] = {}
    for k in sorted(keys):
        present = [a[k] for a in agg_list if k in a]
        if not present:
            continue
        avg_mean = sum(p["mean"] for p in present) / len(present)
        avg_std = math.sqrt(sum(p["std"] ** 2 for p in present) / len(present))
        n = max(p["n_seeds"] for p in present)
        out[k] = {"mean": avg_mean, "std": avg_std, "n_seeds": n}
    return out


def _flatten_metrics(raw_metrics: dict[str, Any]) -> dict[str, float]:
    """Normalize metrics dict to flat {name: float}.

    Handles both flat values (``{mse: 0.4}``) and pre-aggregated dicts
    (``{mse: {mean: 0.4, std: 0.0}}``).
    """
    out: dict[str, float] = {}
    for k, v in raw_metrics.items():
        if isinstance(v, dict):
            # Pre-aggregated — use the mean value
            out[k] = v.get("mean", 0.0)
        else:
            out[k] = float(v)
    return out


def _ingest_entry_yaml(entries_dir: pathlib.Path) -> list[dict[str, Any]]:
    """Load legacy leaderboard/entries/*.yaml (paper baselines)."""
    entries = []
    if not entries_dir.exists():
        return entries
    for p in sorted(entries_dir.glob("*.yaml")):
        try:
            raw = yaml.safe_load(p.read_text()) or []
            for item in raw:
                source = item.get("source", {})
                entries.append({
                    "model": item["model"],
                    "task": item["task"],
                    "dataset": item["dataset"],
                    "seed": None,
                    "hparams": item.get("hparams", {}),
                    "metrics": _flatten_metrics(item.get("metrics", {})),
                    "source_type": source.get("source_type", item.get("source_type", "paper")),
                    "citation": source.get("citation", item.get("citation", "")),
                    "url": source.get("url", item.get("url", "")),
                })
        except Exception as exc:
            print(f"Warning: skipping {p.name}: {exc}")
    return entries


def _build_view(
    view_cfg: dict[str, Any],
    leaderboard_results_dir: pathlib.Path,
    results_dir: pathlib.Path,
    paper_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    task = view_cfg["task"]
    datasets = view_cfg["datasets"]
    variants_cfg = view_cfg["variants"]
    primary_metrics = view_cfg["primary_metrics"]

    first_variant = variants_cfg[0]
    subcolumn_labels = [sc["label"] for sc in first_variant.get("subcolumns", [])]

    all_model_results: dict[str, dict] = {}

    for variant_cfg in variants_cfg:
        vlabel = variant_cfg["label"]
        vmatch = variant_cfg.get("match", {})
        subcolumns = variant_cfg.get("subcolumns", [])

        for dataset in datasets:
            raw = collect_results(leaderboard_results_dir, results_dir, task, dataset)
            raw = [r for r in raw if matches_hparams(r, vmatch)]

            by_model: dict[str, list[dict]] = {}
            for r in raw:
                m = r.get("model", "Unknown")
                by_model.setdefault(m, []).append(r)

            # Add paper entries only for models that have no local-run data for this dataset/variant
            local_run_models = set(by_model.keys())
            for e in paper_entries:
                if (e["task"] == task and e["dataset"] == dataset
                        and matches_hparams(e, vmatch)
                        and e.get("model", "Unknown") not in local_run_models):
                    m = e.get("model", "Unknown")
                    by_model.setdefault(m, []).append(e)

            for model, seeds in by_model.items():
                if model not in all_model_results:
                    first = seeds[0]
                    all_model_results[model] = {
                        "name": model,
                        "source_type": first.get("source_type", "local_run"),
                        "citation": first.get("citation", ""),
                        "url": first.get("url", ""),
                        "results": {},
                    }
                mr = all_model_results[model]["results"]
                mr.setdefault(vlabel, {})
                mr[vlabel].setdefault(dataset, {})

                if subcolumns:
                    subcol_aggs = []
                    for sc in subcolumns:
                        matching = [r for r in seeds if matches_hparams(r, sc["match"])]
                        if matching:
                            agg = aggregate_metrics(matching)
                            mr[vlabel][dataset][sc["label"]] = agg
                            subcol_aggs.append(agg)
                    if subcol_aggs:
                        mr[vlabel][dataset]["avg"] = mean_of_agg(subcol_aggs)
                else:
                    if seeds:
                        mr[vlabel][dataset]["avg"] = aggregate_metrics(seeds)

    return {
        "id": view_cfg["id"],
        "display_name": view_cfg["display_name"],
        "primary_metrics": primary_metrics,
        "variants": [v["label"] for v in variants_cfg],
        "datasets": datasets,
        "subcolumns": subcolumn_labels,
        "models": list(all_model_results.values()),
    }


def build(cfg: dict[str, Any]) -> None:
    lr_dir = pathlib.Path(cfg["leaderboard_results_dir"])
    results_dir = pathlib.Path(cfg["results_dir"])
    views_dir = pathlib.Path(cfg["views_dir"])
    entries_dir = pathlib.Path(cfg["entries_dir"])
    out_path = pathlib.Path(cfg["out"])

    paper_entries = _ingest_entry_yaml(entries_dir)

    views = []
    if views_dir.exists():
        for vf in sorted(views_dir.glob("*.yaml")):
            try:
                view_cfg = yaml.safe_load(vf.read_text())
                views.append(_build_view(view_cfg, lr_dir, results_dir, paper_entries))
            except Exception as exc:
                print(f"Warning: skipping view {vf.name}: {exc}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "views": views,
        "schema": {
            "views": [v["id"] for v in views],
            "datasets_by_view": {v["id"]: v["datasets"] for v in views},
        },
    }, indent=2))
    total_models = sum(len(v["models"]) for v in views)
    print(f"Wrote {len(views)} views, {total_models} model entries → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="leaderboard.yaml", type=pathlib.Path)
    args = parser.parse_args()
    cfg_path = ROOT / args.config
    if not cfg_path.exists():
        build({
            "leaderboard_results_dir": str(ROOT / "leaderboard/results"),
            "results_dir": str(ROOT / "results"),
            "views_dir": str(ROOT / "leaderboard/views"),
            "entries_dir": str(ROOT / "leaderboard/entries"),
            "out": str(ROOT / "leaderboard/webapp/public/leaderboard_data.json"),
        })
    else:
        build(load_config(cfg_path))
