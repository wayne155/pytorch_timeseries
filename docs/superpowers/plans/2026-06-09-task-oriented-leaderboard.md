# Task-Oriented Leaderboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat-list leaderboard with a task-view-oriented design — views defined by YAML, results in `leaderboard_results/`, multi-level column headers (avg + subcolumns per pred_len or mask_rate), and a left-side dataset selector panel.

**Architecture:** The Python build script reads `leaderboard.yaml` config + `leaderboard/views/*.yaml` display configs, scans `leaderboard_results/` (new dir-tree format) and `results/` (legacy flat JSON), aggregates per-model across seeds, and emits a new `leaderboard_data.json`. The React app replaces the filter-sidebar + flat table with ViewSelector tabs, VariantBar, DatasetSelector panel, and TaskTable with TanStack column groups.

**Tech Stack:** Python 3.8+, PyYAML; React 18, TypeScript 5, TanStack Table v8, Tailwind CSS, Vitest.

**Spec:** `docs/superpowers/specs/2026-06-09-task-oriented-leaderboard-design.md`

---

## File Structure

**New Python files:**
- `leaderboard.yaml` — project-root config (paths + remote backend placeholder)
- `leaderboard/views/long_term_forecast.yaml` — view YAML
- `leaderboard/views/short_term_forecast.yaml`
- `leaderboard/views/imputation.yaml`
- `leaderboard/views/anomaly_detection.yaml`
- `leaderboard/views/uea_classification.yaml`
- `leaderboard/views/irregular_classification.yaml`
- `leaderboard_results/.gitkeep` — empty benchmark results dir
- `tests/leaderboard/test_build_task_view.py` — new tests for new build logic

**Modified Python:**
- `scripts/build_leaderboard.py` — complete rewrite (reads views, new JSON format)
- `tests/leaderboard/test_build_script.py` — update to match new build() API

**New TypeScript/React:**
- `webapp/src/hooks/useTaskView.ts` — selects active view/variant/dataset, returns rows + column defs
- `webapp/src/hooks/useTaskView.test.ts`
- `webapp/src/components/DatasetSelector.tsx` — left-side radio panel
- `webapp/src/components/DatasetSelector.test.tsx`
- `webapp/src/components/TaskTable.tsx` — multi-level header TanStack table
- `webapp/src/components/TaskTable.test.tsx`
- `webapp/src/components/ViewSelector.tsx` — tab row for view switching
- `webapp/src/components/VariantBar.tsx` — variant dropdown + std toggle + export
- `webapp/src/components/RowActionButton.tsx` — `[⋯]` placeholder button

**Modified TypeScript:**
- `webapp/src/types.ts` — replace old types with new task-view types
- `webapp/src/hooks/useLeaderboard.ts` — update return type reference
- `webapp/src/App.tsx` — complete rewrite with new layout

**Deleted:**
- `webapp/src/components/FilterSidebar.tsx`
- `webapp/src/components/ToolBar.tsx`
- `webapp/src/hooks/useFiltered.ts`
- `webapp/src/hooks/useFiltered.test.ts`

**Kept unchanged:**
- `webapp/src/components/MetricCell.tsx` + `MetricCell.test.tsx`
- `webapp/src/components/HparamBadges.tsx`

---

## Task 1: Config files — leaderboard.yaml + view YAMLs

**Files:**
- Create: `leaderboard.yaml`
- Create: `leaderboard/views/long_term_forecast.yaml`
- Create: `leaderboard/views/short_term_forecast.yaml`
- Create: `leaderboard/views/imputation.yaml`
- Create: `leaderboard/views/anomaly_detection.yaml`
- Create: `leaderboard/views/uea_classification.yaml`
- Create: `leaderboard/views/irregular_classification.yaml`
- Create: `leaderboard_results/.gitkeep`

- [ ] **Step 1: Create leaderboard.yaml**

```yaml
# leaderboard.yaml  (project root)
leaderboard_results_dir: leaderboard_results
results_dir: results
views_dir: leaderboard/views
entries_dir: leaderboard/entries
out: webapp/public/leaderboard_data.json

# Remote backend — placeholder, not implemented
# backends:
#   - type: remote
#     url: TODO
```

- [ ] **Step 2: Create leaderboard/views/long_term_forecast.yaml**

```yaml
id: long_term_forecast
display_name: "Long-Term Forecast"
task: Forecast
primary_metrics: [mse, mae]

variants:
  - label: "I96"
    match: {windows: 96}
    subcolumns:
      - {label: "96",  match: {pred_len: 96}}
      - {label: "192", match: {pred_len: 192}}
      - {label: "336", match: {pred_len: 336}}
      - {label: "720", match: {pred_len: 720}}
  - label: "I336"
    match: {windows: 336}
    subcolumns:
      - {label: "96",  match: {pred_len: 96}}
      - {label: "192", match: {pred_len: 192}}
      - {label: "336", match: {pred_len: 336}}
      - {label: "720", match: {pred_len: 720}}

datasets:
  - ETTh1
  - ETTh2
  - ETTm1
  - ETTm2
  - Electricity
  - Weather
  - Traffic
  - ExchangeRate
```

- [ ] **Step 3: Create leaderboard/views/short_term_forecast.yaml**

```yaml
id: short_term_forecast
display_name: "Short-Term Forecast"
task: ProbForecast
primary_metrics: [mase, mape]

variants:
  - label: "I96"
    match: {seq_len: 96}
    subcolumns:
      - {label: "24", match: {pred_len: 24}}
      - {label: "48", match: {pred_len: 48}}
      - {label: "96", match: {pred_len: 96}}

datasets:
  - ETTh1
  - ETTh2
  - ETTm1
  - ETTm2
```

- [ ] **Step 4: Create leaderboard/views/imputation.yaml**

```yaml
id: imputation
display_name: "Imputation"
task: Imputation
primary_metrics: [mse, mae]

variants:
  - label: "W96"
    match: {windows: 96}
    subcolumns:
      - {label: "0.125", match: {mask_rate: 0.125}}
      - {label: "0.25",  match: {mask_rate: 0.25}}
      - {label: "0.375", match: {mask_rate: 0.375}}
      - {label: "0.5",   match: {mask_rate: 0.5}}

datasets:
  - ETTh1
  - ETTh2
  - ETTm1
  - ETTm2
  - Electricity
  - Weather
```

- [ ] **Step 5: Create leaderboard/views/anomaly_detection.yaml**

```yaml
id: anomaly_detection
display_name: "Anomaly Detection"
task: AnomalyDetection
primary_metrics: [f1, precision, recall]

variants:
  - label: "W96"
    match: {windows: 96}
    subcolumns: []

datasets:
  - SMD
  - MSL
  - SMAP
  - SWaT
  - PSM
```

- [ ] **Step 6: Create leaderboard/views/uea_classification.yaml**

```yaml
id: uea_classification
display_name: "UEA Classification"
task: UEAClassification
primary_metrics: [accuracy]

variants:
  - label: "W96"
    match: {windows: 96}
    subcolumns: []

datasets:
  - EthanolConcentration
  - FaceDetection
  - Handwriting
  - Heartbeat
  - JapaneseVowels
  - PEMS-SF
  - SelfRegulationSCP1
  - SelfRegulationSCP2
  - SpokenArabicDigits
  - UWaveGestureLibrary
```

- [ ] **Step 7: Create leaderboard/views/irregular_classification.yaml**

```yaml
id: irregular_classification
display_name: "Irregular Classification"
task: IrregularClassification
primary_metrics: [accuracy]

variants:
  - label: "Default"
    match: {}
    subcolumns: []

datasets:
  - PhysioNet2012
  - PhysioNet2019
```

- [ ] **Step 8: Create leaderboard_results/.gitkeep**

```bash
mkdir -p leaderboard_results
touch leaderboard_results/.gitkeep
```

- [ ] **Step 9: Commit**

```bash
git add leaderboard.yaml leaderboard/views/ leaderboard_results/.gitkeep
git commit -m "feat: add leaderboard.yaml config and task view YAMLs"
```

---

## Task 2: Rewrite build script + tests

**Files:**
- Modify: `scripts/build_leaderboard.py`
- Create: `tests/leaderboard/test_build_task_view.py`
- Modify: `tests/leaderboard/test_build_script.py`

- [ ] **Step 1: Write failing tests in test_build_task_view.py**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /data/yww/notebook/pytorchtimseries
python -m pytest tests/leaderboard/test_build_task_view.py -v 2>&1 | head -30
```

Expected: ImportError or AttributeError — functions not yet defined.

- [ ] **Step 3: Rewrite scripts/build_leaderboard.py**

```python
#!/usr/bin/env python3
"""
Build leaderboard_data.json from leaderboard_results/, results/, and view YAMLs.

Usage:
    python scripts/build_leaderboard.py [--config leaderboard.yaml]
"""
from __future__ import annotations

import argparse
import json
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
        "out": str(ROOT / raw.get("out", "webapp/public/leaderboard_data.json")),
    }


def collect_results(
    leaderboard_results_dir: pathlib.Path,
    results_dir: pathlib.Path,
    task: str,
    dataset: str,
) -> list[dict[str, Any]]:
    """Collect per-seed metric dicts for (task, dataset). leaderboard_results takes priority."""
    seen: dict[tuple, dict] = {}

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
                    data = json.loads(mf.read_text())
                    key = (data.get("model"), data.get("seed"))
                    seen[key] = data

    # Priority 2: results/*.json flat format (fallback)
    if results_dir.exists():
        for p in sorted(results_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text())
                if data.get("task") == task and data.get("dataset") == dataset:
                    key = (data.get("model"), data.get("seed"))
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
        avg_std = sum(p["std"] for p in present) / len(present)
        n = max(p["n_seeds"] for p in present)
        out[k] = {"mean": avg_mean, "std": avg_std, "n_seeds": n}
    return out


def _ingest_entry_yaml(entries_dir: pathlib.Path) -> list[dict[str, Any]]:
    """Load legacy leaderboard/entries/*.yaml (paper baselines)."""
    entries = []
    if not entries_dir.exists():
        return entries
    for p in sorted(entries_dir.glob("*.yaml")):
        raw = yaml.safe_load(p.read_text()) or []
        for item in raw:
            entries.append({
                "model": item["model"],
                "task": item["task"],
                "dataset": item["dataset"],
                "seed": None,
                "hparams": item.get("hparams", {}),
                "metrics": item.get("metrics", {}),
                "source_type": "paper",
                "citation": item.get("citation", ""),
                "url": item.get("url", ""),
            })
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

    # subcolumn labels — same for all variants
    first_variant = variants_cfg[0]
    subcolumn_labels = [sc["label"] for sc in first_variant.get("subcolumns", [])]

    all_model_results: dict[str, dict] = {}  # model -> {source_type, citation, url, results}

    for variant_cfg in variants_cfg:
        vlabel = variant_cfg["label"]
        vmatch = variant_cfg.get("match", {})
        subcolumns = variant_cfg.get("subcolumns", [])

        for dataset in datasets:
            raw = collect_results(leaderboard_results_dir, results_dir, task, dataset)
            # Filter by variant match
            raw = [r for r in raw if matches_hparams(r, vmatch)]

            # Add matching paper entries
            for e in paper_entries:
                if e["task"] == task and e["dataset"] == dataset and matches_hparams(e, vmatch):
                    raw.append(e)

            # Group by model
            by_model: dict[str, list[dict]] = {}
            for r in raw:
                m = r.get("model", "Unknown")
                by_model.setdefault(m, []).append(r)

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
                    # No subcolumns: avg = direct aggregate over all seeds
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
        # Fallback defaults for backward compat
        build({
            "leaderboard_results_dir": str(ROOT / "leaderboard_results"),
            "results_dir": str(ROOT / "results"),
            "views_dir": str(ROOT / "leaderboard/views"),
            "entries_dir": str(ROOT / "leaderboard/entries"),
            "out": str(ROOT / "webapp/public/leaderboard_data.json"),
        })
    else:
        build(load_config(cfg_path))
```

- [ ] **Step 4: Update tests/leaderboard/test_build_script.py to remove stale API calls**

The old tests called `build(results_dir, entries_dir, out)` with positional args. Replace the `test_build_*` tests that call `build()` with new signatures, keeping the `make_id`, `ingest_result_json`, `ingest_yaml_entries` tests (those functions are removed — delete the whole file and replace with a minimal smoke test).

Delete `tests/leaderboard/test_build_script.py` and replace with:

```python
# tests/leaderboard/test_build_script.py
"""Smoke tests for the build script CLI entrypoint."""
from __future__ import annotations
import json
import pathlib
import subprocess
import sys


def test_build_script_runs(tmp_path):
    """Build script runs without error when dirs are empty."""
    cfg = {
        "leaderboard_results_dir": str(tmp_path / "lr"),
        "results_dir": str(tmp_path / "results"),
        "views_dir": str(tmp_path / "views"),
        "entries_dir": str(tmp_path / "entries"),
        "out": str(tmp_path / "out.json"),
    }
    import yaml
    cfg_path = tmp_path / "leaderboard.yaml"
    cfg_path.write_text(yaml.dump({
        k: v for k, v in cfg.items()
    }))
    result = subprocess.run(
        [sys.executable, "scripts/build_leaderboard.py", "--config", str(cfg_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads((tmp_path / "out.json").read_text())
    assert "views" in data
    assert "generated_at" in data
```

- [ ] **Step 5: Run all leaderboard tests**

```bash
cd /data/yww/notebook/pytorchtimseries
python -m pytest tests/leaderboard/ -v
```

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_leaderboard.py tests/leaderboard/
git commit -m "feat: rewrite build script for task-view format with leaderboard_results/ support"
```

---

## Task 3: TypeScript types

**Files:**
- Modify: `webapp/src/types.ts`
- Modify: `webapp/src/hooks/useLeaderboard.ts`

- [ ] **Step 1: Replace webapp/src/types.ts**

```typescript
// webapp/src/types.ts

export interface SubcolumnMetrics {
  [metric: string]: { mean: number; std: number; n_seeds: number }
}

export interface DatasetResult {
  avg: SubcolumnMetrics
  [subcolumn: string]: SubcolumnMetrics
}

export interface ModelResult {
  name: string
  source_type: 'local_run' | 'paper'
  citation: string
  url: string
  results: {
    [variant: string]: {
      [dataset: string]: DatasetResult
    }
  }
}

export interface ViewData {
  id: string
  display_name: string
  primary_metrics: string[]
  variants: string[]
  datasets: string[]
  subcolumns: string[]
  models: ModelResult[]
}

export interface TaskLeaderboardData {
  generated_at: string
  views: ViewData[]
  schema: {
    views: string[]
    datasets_by_view: Record<string, string[]>
  }
}

/** One display row in TaskTable. */
export interface TaskTableRow {
  model: string
  source_type: string
  citation: string
  url: string
  /** colId → aggregated metrics. colId is "avg" | subcolumn label | dataset name */
  columns: Record<string, SubcolumnMetrics | null>
}

export interface TaskViewOptions {
  showStd: boolean
  sortColumn: string | null
  sortMetric: string | null
  sortDirection: 'asc' | 'desc'
}
```

- [ ] **Step 2: Update webapp/src/hooks/useLeaderboard.ts**

```typescript
import { useState, useEffect } from 'react'
import type { TaskLeaderboardData } from '../types'

export function useLeaderboard(url = './leaderboard_data.json') {
  const [data, setData] = useState<TaskLeaderboardData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const load = () => {
    setLoading(true)
    setError(null)
    fetch(url)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json() as Promise<TaskLeaderboardData>
      })
      .then(d => { setData(d); setLoading(false) })
      .catch((e: Error) => { setError(e.message); setLoading(false) })
  }

  useEffect(() => { load() }, [url])
  return { data, error, loading, refresh: load }
}
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd /data/yww/notebook/pytorchtimseries/webapp
npx tsc --noEmit 2>&1 | head -30
```

Expected: Errors only in old files (FilterSidebar, useFiltered, App.tsx, LeaderboardTable) that reference old types. These will be fixed in later tasks. Types themselves should be clean.

- [ ] **Step 4: Commit**

```bash
git add webapp/src/types.ts webapp/src/hooks/useLeaderboard.ts
git commit -m "feat: new TypeScript types for task-view leaderboard format"
```

---

## Task 4: useTaskView hook + tests

**Files:**
- Create: `webapp/src/hooks/useTaskView.ts`
- Create: `webapp/src/hooks/useTaskView.test.ts`

- [ ] **Step 1: Write failing tests**

```typescript
// webapp/src/hooks/useTaskView.test.ts
import { describe, it, expect } from 'vitest'
import { renderHook } from '@testing-library/react'
import { useTaskView } from './useTaskView'
import type { ViewData, TaskViewOptions } from '../types'

const DEFAULT_OPTS: TaskViewOptions = {
  showStd: false, sortColumn: null, sortMetric: null, sortDirection: 'asc',
}

function makeView(overrides: Partial<ViewData> = {}): ViewData {
  return {
    id: 'long_term_forecast',
    display_name: 'Long-Term Forecast',
    primary_metrics: ['mse', 'mae'],
    variants: ['I96'],
    datasets: ['ETTh1', 'ETTh2'],
    subcolumns: ['96', '192'],
    models: [
      {
        name: 'DLinear',
        source_type: 'local_run',
        citation: '',
        url: '',
        results: {
          I96: {
            ETTh1: {
              avg: { mse: { mean: 0.417, std: 0.008, n_seeds: 3 }, mae: { mean: 0.300, std: 0.004, n_seeds: 3 } },
              '96':  { mse: { mean: 0.384, std: 0.007, n_seeds: 3 }, mae: { mean: 0.276, std: 0.003, n_seeds: 3 } },
              '192': { mse: { mean: 0.450, std: 0.009, n_seeds: 3 }, mae: { mean: 0.324, std: 0.005, n_seeds: 3 } },
            },
            ETTh2: {
              avg: { mse: { mean: 0.380, std: 0.005, n_seeds: 3 }, mae: { mean: 0.290, std: 0.003, n_seeds: 3 } },
              '96':  { mse: { mean: 0.350, std: 0.005, n_seeds: 3 }, mae: { mean: 0.265, std: 0.002, n_seeds: 3 } },
              '192': { mse: { mean: 0.410, std: 0.006, n_seeds: 3 }, mae: { mean: 0.315, std: 0.004, n_seeds: 3 } },
            },
          },
        },
      },
    ],
    ...overrides,
  }
}

describe('useTaskView — with subcolumns (Forecast)', () => {
  it('returns one row per model when dataset is specific', () => {
    const view = makeView()
    const { result } = renderHook(() =>
      useTaskView(view, 'I96', 'ETTh1', DEFAULT_OPTS)
    )
    expect(result.current.rows).toHaveLength(1)
    const row = result.current.rows[0]
    expect(row.model).toBe('DLinear')
    expect(row.columns['avg']?.mse.mean).toBeCloseTo(0.417)
    expect(row.columns['96']?.mse.mean).toBeCloseTo(0.384)
    expect(row.columns['192']?.mse.mean).toBeCloseTo(0.450)
  })

  it('returns avg-across-datasets when dataset is All', () => {
    const view = makeView()
    const { result } = renderHook(() =>
      useTaskView(view, 'I96', 'All', DEFAULT_OPTS)
    )
    const row = result.current.rows[0]
    // avg across ETTh1 (0.417) and ETTh2 (0.380) → 0.3985
    expect(row.columns['avg']?.mse.mean).toBeCloseTo(0.3985)
    expect(row.columns['96']?.mse.mean).toBeCloseTo((0.384 + 0.350) / 2)
  })

  it('columnDefs includes avg + subcolumns when dataset is specific', () => {
    const view = makeView()
    const { result } = renderHook(() =>
      useTaskView(view, 'I96', 'ETTh1', DEFAULT_OPTS)
    )
    const ids = result.current.columnDefs.map(c => c.id)
    expect(ids).toContain('avg')
    expect(ids).toContain('96')
    expect(ids).toContain('192')
  })
})

describe('useTaskView — no subcolumns (UEA)', () => {
  it('returns one column per dataset when All selected', () => {
    const uea = makeView({
      id: 'uea_classification',
      subcolumns: [],
      primary_metrics: ['accuracy'],
      models: [{
        name: 'GRU-D', source_type: 'local_run', citation: '', url: '',
        results: {
          W96: {
            EthanolConcentration: { avg: { accuracy: { mean: 0.71, std: 0.02, n_seeds: 3 } } },
            FaceDetection:        { avg: { accuracy: { mean: 0.75, std: 0.01, n_seeds: 3 } } },
          },
        },
      }],
      variants: ['W96'],
      datasets: ['EthanolConcentration', 'FaceDetection'],
    })
    const { result } = renderHook(() =>
      useTaskView(uea, 'W96', 'All', DEFAULT_OPTS)
    )
    const ids = result.current.columnDefs.map(c => c.id)
    expect(ids).toContain('avg')
    expect(ids).toContain('EthanolConcentration')
    expect(ids).toContain('FaceDetection')
    const row = result.current.rows[0]
    expect(row.columns['EthanolConcentration']?.accuracy.mean).toBeCloseTo(0.71)
  })

  it('returns only avg column when specific dataset selected', () => {
    const uea = makeView({
      id: 'uea_classification',
      subcolumns: [],
      primary_metrics: ['accuracy'],
      models: [{
        name: 'GRU-D', source_type: 'local_run', citation: '', url: '',
        results: {
          W96: {
            EthanolConcentration: { avg: { accuracy: { mean: 0.71, std: 0.02, n_seeds: 3 } } },
          },
        },
      }],
      variants: ['W96'],
      datasets: ['EthanolConcentration'],
    })
    const { result } = renderHook(() =>
      useTaskView(uea, 'W96', 'EthanolConcentration', DEFAULT_OPTS)
    )
    const ids = result.current.columnDefs.map(c => c.id)
    expect(ids).toEqual(['avg'])
  })
})

describe('useTaskView — sorting', () => {
  it('sorts rows by avg mse ascending', () => {
    const view = makeView({
      models: [
        { name: 'Worse', source_type: 'local_run', citation: '', url: '',
          results: { I96: { ETTh1: { avg: { mse: { mean: 0.50, std: 0.01, n_seeds: 3 } } } } } },
        { name: 'Better', source_type: 'local_run', citation: '', url: '',
          results: { I96: { ETTh1: { avg: { mse: { mean: 0.38, std: 0.01, n_seeds: 3 } } } } } },
      ],
    })
    const opts = { ...DEFAULT_OPTS, sortColumn: 'avg', sortMetric: 'mse', sortDirection: 'asc' as const }
    const { result } = renderHook(() => useTaskView(view, 'I96', 'ETTh1', opts))
    expect(result.current.rows[0].model).toBe('Better')
    expect(result.current.rows[1].model).toBe('Worse')
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /data/yww/notebook/pytorchtimseries/webapp
npx vitest run src/hooks/useTaskView.test.ts 2>&1 | head -20
```

Expected: Error — cannot find module './useTaskView'.

- [ ] **Step 3: Create webapp/src/hooks/useTaskView.ts**

```typescript
// webapp/src/hooks/useTaskView.ts
import { useMemo } from 'react'
import type { ViewData, TaskTableRow, TaskViewOptions, SubcolumnMetrics } from '../types'

const LOWER_BETTER = ['mse', 'mae', 'loss', 'error', 'rmse', 'crps', 'wis', 'nll']

export function isLowerBetter(key: string): boolean {
  const k = key.toLowerCase()
  return LOWER_BETTER.some(p => k.includes(p))
}

function meanOfSubcolMetrics(
  subcols: SubcolumnMetrics[],
): SubcolumnMetrics {
  if (subcols.length === 0) return {}
  const keys = [...new Set(subcols.flatMap(s => Object.keys(s)))]
  const out: SubcolumnMetrics = {}
  for (const k of keys) {
    const present = subcols.filter(s => k in s)
    if (!present.length) continue
    const meanVal = present.reduce((s, p) => s + p[k].mean, 0) / present.length
    const stdVal = present.reduce((s, p) => s + p[k].std, 0) / present.length
    const nSeeds = Math.max(...present.map(p => p[k].n_seeds))
    out[k] = { mean: meanVal, std: stdVal, n_seeds: nSeeds }
  }
  return out
}

export interface ColumnDef {
  id: string
  label: string
}

export interface TaskViewResult {
  rows: TaskTableRow[]
  columnDefs: ColumnDef[]
}

export function useTaskView(
  view: ViewData | null,
  variant: string,
  selectedDataset: string,
  options: TaskViewOptions,
): TaskViewResult {
  return useMemo(() => {
    if (!view) return { rows: [], columnDefs: [] }

    const hasSubcolumns = view.subcolumns.length > 0

    // Build column defs
    let columnDefs: ColumnDef[]
    if (hasSubcolumns) {
      // avg + each subcolumn — same regardless of dataset selection
      columnDefs = [
        { id: 'avg', label: 'avg' },
        ...view.subcolumns.map(sc => ({ id: sc, label: sc })),
      ]
    } else if (selectedDataset === 'All') {
      // avg + one per dataset
      columnDefs = [
        { id: 'avg', label: 'avg' },
        ...view.datasets.map(d => ({ id: d, label: d })),
      ]
    } else {
      // just avg for the selected dataset
      columnDefs = [{ id: 'avg', label: 'avg' }]
    }

    // Build rows
    const rows: TaskTableRow[] = view.models.map(model => {
      const variantResults = model.results[variant] ?? {}
      const columns: Record<string, SubcolumnMetrics | null> = {}

      if (hasSubcolumns) {
        const datasetsToAvg = selectedDataset === 'All'
          ? view.datasets
          : [selectedDataset]

        // avg across datasets for each subcolumn
        for (const sc of view.subcolumns) {
          const subcolAggs = datasetsToAvg
            .map(d => variantResults[d]?.[sc])
            .filter((x): x is SubcolumnMetrics => x != null)
          columns[sc] = subcolAggs.length > 0 ? meanOfSubcolMetrics(subcolAggs) : null
        }
        // overall avg = mean of subcolumn avgs
        const subcolValues = view.subcolumns
          .map(sc => columns[sc])
          .filter((x): x is SubcolumnMetrics => x != null)
        columns['avg'] = subcolValues.length > 0 ? meanOfSubcolMetrics(subcolValues) : null

      } else if (selectedDataset === 'All') {
        // one column per dataset
        let avgInputs: SubcolumnMetrics[] = []
        for (const d of view.datasets) {
          const val = variantResults[d]?.avg ?? null
          columns[d] = val
          if (val) avgInputs.push(val)
        }
        columns['avg'] = avgInputs.length > 0 ? meanOfSubcolMetrics(avgInputs) : null
      } else {
        columns['avg'] = variantResults[selectedDataset]?.avg ?? null
      }

      return {
        model: model.name,
        source_type: model.source_type,
        citation: model.citation,
        url: model.url,
        columns,
      }
    })

    // Sort
    let sorted = rows
    if (options.sortColumn && options.sortMetric) {
      const col = options.sortColumn
      const metric = options.sortMetric
      const dir = options.sortDirection === 'asc' ? 1 : -1
      sorted = [...rows].sort((a, b) => {
        const av = a.columns[col]?.[metric]?.mean ?? (isLowerBetter(metric) ? Infinity : -Infinity)
        const bv = b.columns[col]?.[metric]?.mean ?? (isLowerBetter(metric) ? Infinity : -Infinity)
        return dir * (av - bv)
      })
    }

    return { rows: sorted, columnDefs }
  }, [view, variant, selectedDataset, options])
}
```

- [ ] **Step 4: Run tests**

```bash
cd /data/yww/notebook/pytorchtimseries/webapp
npx vitest run src/hooks/useTaskView.test.ts
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add webapp/src/hooks/useTaskView.ts webapp/src/hooks/useTaskView.test.ts
git commit -m "feat: useTaskView hook with subcolumn/dataset-selector logic"
```

---

## Task 5: DatasetSelector component + tests

**Files:**
- Create: `webapp/src/components/DatasetSelector.tsx`
- Create: `webapp/src/components/DatasetSelector.test.tsx`

- [ ] **Step 1: Write failing tests**

```tsx
// webapp/src/components/DatasetSelector.test.tsx
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { DatasetSelector } from './DatasetSelector'

const DATASETS = ['ETTh1', 'ETTh2', 'ETTm1']

describe('DatasetSelector', () => {
  it('renders All option and all datasets', () => {
    render(
      <DatasetSelector datasets={DATASETS} selected="All" onChange={() => {}} />
    )
    expect(screen.getByLabelText('All')).toBeInTheDocument()
    DATASETS.forEach(d => expect(screen.getByLabelText(d)).toBeInTheDocument())
  })

  it('marks the selected dataset as checked', () => {
    render(
      <DatasetSelector datasets={DATASETS} selected="ETTh2" onChange={() => {}} />
    )
    expect(screen.getByLabelText('ETTh2')).toBeChecked()
    expect(screen.getByLabelText('All')).not.toBeChecked()
  })

  it('calls onChange when a dataset is clicked', () => {
    const onChange = vi.fn()
    render(
      <DatasetSelector datasets={DATASETS} selected="All" onChange={onChange} />
    )
    fireEvent.click(screen.getByLabelText('ETTh1'))
    expect(onChange).toHaveBeenCalledWith('ETTh1')
  })

  it('calls onChange with All when All is clicked', () => {
    const onChange = vi.fn()
    render(
      <DatasetSelector datasets={DATASETS} selected="ETTh1" onChange={onChange} />
    )
    fireEvent.click(screen.getByLabelText('All'))
    expect(onChange).toHaveBeenCalledWith('All')
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /data/yww/notebook/pytorchtimseries/webapp
npx vitest run src/components/DatasetSelector.test.tsx 2>&1 | head -10
```

Expected: Cannot find module.

- [ ] **Step 3: Create DatasetSelector.tsx**

```tsx
// webapp/src/components/DatasetSelector.tsx

interface DatasetSelectorProps {
  datasets: string[]
  selected: string
  onChange: (dataset: string) => void
}

export function DatasetSelector({ datasets, selected, onChange }: DatasetSelectorProps) {
  const options = ['All', ...datasets]
  return (
    <aside className="w-44 shrink-0 bg-white border-r border-gray-200 overflow-y-auto">
      <div className="px-3 pt-3 pb-1 text-xs font-semibold text-gray-500 uppercase tracking-wide">
        Datasets
      </div>
      <div className="pb-3">
        {options.map(opt => (
          <label
            key={opt}
            className="flex items-center gap-2 px-3 py-1 text-sm cursor-pointer hover:bg-gray-50"
          >
            <input
              type="radio"
              name="dataset-selector"
              checked={selected === opt}
              onChange={() => onChange(opt)}
              className="accent-blue-600"
            />
            <span className={`truncate ${selected === opt ? 'font-medium text-gray-900' : 'text-gray-700'}`}>
              {opt}
            </span>
          </label>
        ))}
      </div>
    </aside>
  )
}
```

- [ ] **Step 4: Run tests**

```bash
cd /data/yww/notebook/pytorchtimseries/webapp
npx vitest run src/components/DatasetSelector.test.tsx
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add webapp/src/components/DatasetSelector.tsx webapp/src/components/DatasetSelector.test.tsx
git commit -m "feat: DatasetSelector left-panel radio component"
```

---

## Task 6: TaskTable component + tests

**Files:**
- Create: `webapp/src/components/TaskTable.tsx`
- Create: `webapp/src/components/TaskTable.test.tsx`

- [ ] **Step 1: Write failing tests**

```tsx
// webapp/src/components/TaskTable.test.tsx
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { TaskTable } from './TaskTable'
import type { TaskTableRow, ColumnDef, TaskViewOptions } from '../types'

const DEFAULT_OPTS: TaskViewOptions = {
  showStd: false, sortColumn: null, sortMetric: null, sortDirection: 'asc',
}

function makeRow(name: string, avgMse = 0.4): TaskTableRow {
  return {
    model: name,
    source_type: 'local_run',
    citation: '',
    url: '',
    columns: {
      avg: { mse: { mean: avgMse, std: 0.01, n_seeds: 3 }, mae: { mean: 0.30, std: 0.005, n_seeds: 3 } },
      '96': { mse: { mean: avgMse - 0.03, std: 0.008, n_seeds: 3 }, mae: { mean: 0.27, std: 0.004, n_seeds: 3 } },
    },
  }
}

const COL_DEFS: ColumnDef[] = [
  { id: 'avg', label: 'avg' },
  { id: '96', label: '96' },
]

describe('TaskTable', () => {
  it('renders column group headers (avg, 96)', () => {
    render(
      <TaskTable
        rows={[makeRow('DLinear')]}
        columnDefs={COL_DEFS}
        primaryMetrics={['mse', 'mae']}
        viewOptions={DEFAULT_OPTS}
        onSortChange={() => {}}
      />
    )
    expect(screen.getAllByText('avg').length).toBeGreaterThan(0)
    expect(screen.getAllByText('96').length).toBeGreaterThan(0)
  })

  it('renders metric sub-headers (mse, mae) under each group', () => {
    render(
      <TaskTable
        rows={[makeRow('DLinear')]}
        columnDefs={COL_DEFS}
        primaryMetrics={['mse', 'mae']}
        viewOptions={DEFAULT_OPTS}
        onSortChange={() => {}}
      />
    )
    // mse and mae appear once per column group
    expect(screen.getAllByText(/mse/i).length).toBe(COL_DEFS.length)
    expect(screen.getAllByText(/mae/i).length).toBe(COL_DEFS.length)
  })

  it('renders model name in a row', () => {
    render(
      <TaskTable
        rows={[makeRow('DLinear'), makeRow('PatchTST', 0.35)]}
        columnDefs={COL_DEFS}
        primaryMetrics={['mse']}
        viewOptions={DEFAULT_OPTS}
        onSortChange={() => {}}
      />
    )
    expect(screen.getByText('DLinear')).toBeInTheDocument()
    expect(screen.getByText('PatchTST')).toBeInTheDocument()
  })

  it('shows empty state when no rows', () => {
    render(
      <TaskTable
        rows={[]}
        columnDefs={COL_DEFS}
        primaryMetrics={['mse']}
        viewOptions={DEFAULT_OPTS}
        onSortChange={() => {}}
      />
    )
    expect(screen.getByText(/no results/i)).toBeInTheDocument()
  })

  it('renders [⋯] button per row', () => {
    render(
      <TaskTable
        rows={[makeRow('DLinear'), makeRow('PatchTST')]}
        columnDefs={COL_DEFS}
        primaryMetrics={['mse']}
        viewOptions={DEFAULT_OPTS}
        onSortChange={() => {}}
      />
    )
    expect(screen.getAllByTitle('Row actions')).toHaveLength(2)
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /data/yww/notebook/pytorchtimseries/webapp
npx vitest run src/components/TaskTable.test.tsx 2>&1 | head -10
```

- [ ] **Step 3: Create TaskTable.tsx**

```tsx
// webapp/src/components/TaskTable.tsx
import { useState, useMemo } from 'react'
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table'
import { MetricCell } from './MetricCell'
import { isLowerBetter } from '../hooks/useTaskView'
import type { TaskTableRow, ColumnDef as ColDef, TaskViewOptions, SubcolumnMetrics } from '../types'

interface TaskTableProps {
  rows: TaskTableRow[]
  columnDefs: ColDef[]
  primaryMetrics: string[]
  viewOptions: TaskViewOptions
  onSortChange: (column: string, metric: string, dir: 'asc' | 'desc') => void
}

function getBestWorstForCol(
  rows: TaskTableRow[],
  colId: string,
  metric: string,
): { best: number; worst: number } {
  const vals = rows
    .map(r => r.columns[colId]?.[metric]?.mean)
    .filter((v): v is number => v != null)
  if (!vals.length) return { best: NaN, worst: NaN }
  return isLowerBetter(metric)
    ? { best: Math.min(...vals), worst: Math.max(...vals) }
    : { best: Math.max(...vals), worst: Math.min(...vals) }
}

export function TaskTable({
  rows, columnDefs, primaryMetrics, viewOptions, onSortChange,
}: TaskTableProps) {
  const [sorting, setSorting] = useState<SortingState>([])

  const bestWorst = useMemo(() => {
    const bw: Record<string, Record<string, { best: number; worst: number }>> = {}
    for (const col of columnDefs) {
      bw[col.id] = {}
      for (const metric of primaryMetrics) {
        bw[col.id][metric] = getBestWorstForCol(rows, col.id, metric)
      }
    }
    return bw
  }, [rows, columnDefs, primaryMetrics])

  // Build TanStack column groups: one group per colDef, each with metric leaf columns
  const columns = useMemo<ColumnDef<TaskTableRow>[]>(() => {
    const modelCol: ColumnDef<TaskTableRow> = {
      id: 'model',
      header: 'Model',
      enableSorting: false,
      cell: info => {
        const r = info.row.original
        return r.url
          ? <a href={r.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">{r.model}</a>
          : <span>{r.model}</span>
      },
    }

    const metricGroups: ColumnDef<TaskTableRow>[] = columnDefs.map(col => ({
      id: col.id,
      header: col.label === 'avg'
        ? () => <span className="font-semibold">avg</span>
        : col.label,
      enableSorting: false,
      columns: primaryMetrics.map(metric => ({
        id: `${col.id}__${metric}`,
        header: () => {
          const isSorted = sorting[0]?.id === `${col.id}__${metric}`
          return (
            <span
              className="cursor-pointer select-none hover:text-blue-600 flex items-center gap-0.5"
              title={isLowerBetter(metric) ? 'lower is better' : 'higher is better'}
            >
              {metric} {isLowerBetter(metric) ? '↓' : '↑'}
              {isSorted ? (sorting[0].desc ? ' ▼' : ' ▲') : ''}
            </span>
          )
        },
        accessorFn: (row: TaskTableRow) => row.columns[col.id]?.[metric]?.mean ?? null,
        cell: info => {
          const cellData = info.row.original.columns[col.id]
          if (!cellData || !(metric in cellData)) {
            return <span className="text-gray-300 tabular-nums">—</span>
          }
          const m = cellData[metric]
          const { best, worst } = bestWorst[col.id]?.[metric] ?? { best: NaN, worst: NaN }
          return (
            <MetricCell
              value={{ mean: m.mean, std: m.std }}
              isBest={m.mean === best}
              isWorst={m.mean === worst}
              showStd={viewOptions.showStd}
            />
          )
        },
      } as ColumnDef<TaskTableRow>)),
    } as ColumnDef<TaskTableRow>))

    const actionCol: ColumnDef<TaskTableRow> = {
      id: '_action',
      header: '',
      enableSorting: false,
      size: 40,
      cell: () => (
        <button
          title="Row actions"
          className="text-gray-400 hover:text-gray-600 px-1 rounded text-base leading-none"
          onClick={() => {}}
        >
          ⋯
        </button>
      ),
    }

    return [modelCol, ...metricGroups, actionCol]
  }, [columnDefs, primaryMetrics, bestWorst, viewOptions.showStd, sorting])

  const table = useReactTable({
    data: rows,
    columns,
    state: { sorting },
    onSortingChange: (updater) => {
      const next = typeof updater === 'function' ? updater(sorting) : updater
      setSorting(next)
      if (next.length > 0) {
        const [colId, metric] = next[0].id.split('__')
        if (colId && metric) {
          onSortChange(colId, metric, next[0].desc ? 'desc' : 'asc')
        }
      }
    },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  })

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm text-left border-collapse">
        <thead className="bg-gray-50 border-b border-gray-200 sticky top-0 z-10">
          {table.getHeaderGroups().map(hg => (
            <tr key={hg.id}>
              {hg.headers.map(header => (
                <th
                  key={header.id}
                  colSpan={header.colSpan}
                  style={{ width: header.getSize() }}
                  className={`px-3 py-1.5 text-xs font-semibold text-gray-600 whitespace-nowrap border-r border-gray-100 last:border-r-0 ${
                    header.column.getCanSort() ? 'cursor-pointer select-none hover:bg-gray-100' : ''
                  } ${header.depth === 0 && header.colSpan > 1 ? 'text-center border-b border-gray-200' : ''}`}
                  onClick={header.column.getToggleSortingHandler()}
                >
                  {header.isPlaceholder ? null : flexRender(header.column.columnDef.header, header.getContext())}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody className="divide-y divide-gray-100">
          {table.getRowModel().rows.map((row, i) => (
            <tr key={row.id} className={`hover:bg-gray-50 ${i % 2 === 0 ? '' : 'bg-gray-50/30'}`}>
              {row.getVisibleCells().map(cell => (
                <td key={cell.id} className="px-3 py-1.5 whitespace-nowrap border-r border-gray-50 last:border-r-0">
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td
                colSpan={columns.length * primaryMetrics.length + 2}
                className="px-3 py-8 text-center text-gray-400"
              >
                No results for this view/variant/dataset combination.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
```

- [ ] **Step 4: Run tests**

```bash
cd /data/yww/notebook/pytorchtimseries/webapp
npx vitest run src/components/TaskTable.test.tsx
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add webapp/src/components/TaskTable.tsx webapp/src/components/TaskTable.test.tsx
git commit -m "feat: TaskTable with multi-level column groups (avg + subcolumns)"
```

---

## Task 7: ViewSelector + VariantBar components

**Files:**
- Create: `webapp/src/components/ViewSelector.tsx`
- Create: `webapp/src/components/VariantBar.tsx`

No TDD for these — purely presentational; tested via App integration.

- [ ] **Step 1: Create ViewSelector.tsx**

```tsx
// webapp/src/components/ViewSelector.tsx
import type { ViewData } from '../types'

interface ViewSelectorProps {
  views: ViewData[]
  selectedId: string
  onSelect: (id: string) => void
}

export function ViewSelector({ views, selectedId, onSelect }: ViewSelectorProps) {
  return (
    <div className="flex gap-0 border-b border-gray-200 bg-white px-4 overflow-x-auto">
      {views.map(v => (
        <button
          key={v.id}
          onClick={() => onSelect(v.id)}
          className={`px-4 py-2.5 text-sm font-medium whitespace-nowrap border-b-2 transition-colors ${
            v.id === selectedId
              ? 'border-blue-600 text-blue-700'
              : 'border-transparent text-gray-600 hover:text-gray-900 hover:border-gray-300'
          }`}
        >
          {v.display_name}
        </button>
      ))}
    </div>
  )
}
```

- [ ] **Step 2: Create VariantBar.tsx**

```tsx
// webapp/src/components/VariantBar.tsx
import type { TaskViewOptions } from '../types'

interface VariantBarProps {
  variants: string[]
  selectedVariant: string
  viewOptions: TaskViewOptions
  resultCount: number
  onVariantChange: (v: string) => void
  onOptionsChange: (o: TaskViewOptions) => void
  onExportCsv: () => void
  onRefresh?: () => void
}

export function VariantBar({
  variants, selectedVariant, viewOptions, resultCount,
  onVariantChange, onOptionsChange, onExportCsv, onRefresh,
}: VariantBarProps) {
  return (
    <div className="flex items-center gap-3 px-4 py-2 bg-white border-b border-gray-200 flex-wrap">
      {/* Variant selector */}
      {variants.length > 1 && (
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-gray-500">Variant:</span>
          <select
            className="text-sm border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-blue-400"
            value={selectedVariant}
            onChange={e => onVariantChange(e.target.value)}
          >
            {variants.map(v => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>
      )}

      {/* ±std toggle */}
      <label className="flex items-center gap-1.5 text-sm text-gray-600 cursor-pointer select-none">
        <input
          type="checkbox"
          checked={viewOptions.showStd}
          onChange={e => onOptionsChange({ ...viewOptions, showStd: e.target.checked })}
        />
        ±std
      </label>

      {/* Result count */}
      <span className="text-sm text-gray-400 ml-auto">{resultCount} models</span>

      {/* Export CSV */}
      <button
        className="text-sm px-3 py-1 rounded border border-gray-300 hover:border-gray-400 bg-white text-gray-700"
        onClick={onExportCsv}
      >
        Export CSV
      </button>

      {/* Refresh (live server only) */}
      {onRefresh && (
        <button
          className="text-sm px-3 py-1 rounded border border-gray-300 hover:border-gray-400 bg-white text-gray-700"
          onClick={onRefresh}
        >
          ↺ Refresh
        </button>
      )}
    </div>
  )
}
```

- [ ] **Step 3: Commit**

```bash
git add webapp/src/components/ViewSelector.tsx webapp/src/components/VariantBar.tsx
git commit -m "feat: ViewSelector tab bar and VariantBar controls"
```

---

## Task 8: RowActionButton component

**Files:**
- Create: `webapp/src/components/RowActionButton.tsx`

(Already inlined into TaskTable in Task 6 — this task creates a standalone component for future use, then TaskTable imports it.)

- [ ] **Step 1: Create RowActionButton.tsx**

```tsx
// webapp/src/components/RowActionButton.tsx

interface RowActionButtonProps {
  model: string
  dataset: string
  view: string
}

export function RowActionButton({ model, dataset, view }: RowActionButtonProps) {
  return (
    <button
      title="Row actions"
      aria-label={`Actions for ${model} on ${dataset} (${view})`}
      disabled
      className="text-gray-300 px-1.5 py-0.5 rounded text-base leading-none cursor-not-allowed"
    >
      ⋯
    </button>
  )
}
```

- [ ] **Step 2: Update TaskTable.tsx to use RowActionButton**

In `webapp/src/components/TaskTable.tsx`, replace the inline button in the `actionCol`:

```tsx
// Add import at top:
import { RowActionButton } from './RowActionButton'

// Replace the actionCol cell:
cell: info => (
  <RowActionButton
    model={info.row.original.model}
    dataset=""
    view=""
  />
),
```

- [ ] **Step 3: Commit**

```bash
git add webapp/src/components/RowActionButton.tsx webapp/src/components/TaskTable.tsx
git commit -m "feat: RowActionButton placeholder component"
```

---

## Task 9: App.tsx rewiring + cleanup

**Files:**
- Modify: `webapp/src/App.tsx`
- Delete: `webapp/src/components/FilterSidebar.tsx`
- Delete: `webapp/src/components/ToolBar.tsx`
- Delete: `webapp/src/hooks/useFiltered.ts`
- Delete: `webapp/src/hooks/useFiltered.test.ts`

- [ ] **Step 1: Write new App.tsx**

```tsx
// webapp/src/App.tsx
import { useState, useMemo, useCallback } from 'react'
import { useLeaderboard } from './hooks/useLeaderboard'
import { useTaskView } from './hooks/useTaskView'
import { ViewSelector } from './components/ViewSelector'
import { VariantBar } from './components/VariantBar'
import { DatasetSelector } from './components/DatasetSelector'
import { TaskTable } from './components/TaskTable'
import type { TaskViewOptions } from './types'

const isLiveServer = !import.meta.env.PROD || window.location.port !== ''

function rowsToCsv(rows: ReturnType<typeof useTaskView>['rows'], columnDefs: ReturnType<typeof useTaskView>['columnDefs'], primaryMetrics: string[]): string {
  const metricHeaders = columnDefs.flatMap(c =>
    primaryMetrics.flatMap(m => [`${c.id}_${m}_mean`, `${c.id}_${m}_std`])
  )
  const header = ['model', 'source', ...metricHeaders]
  const lines = rows.map(r => {
    const cells = columnDefs.flatMap(c =>
      primaryMetrics.flatMap(m => {
        const val = r.columns[c.id]?.[m]
        return val ? [String(val.mean), String(val.std)] : ['', '']
      })
    )
    return [r.model, r.source_type, ...cells]
      .map(v => `"${String(v).replace(/"/g, '""')}"`)
      .join(',')
  })
  return [header.join(','), ...lines].join('\n')
}

function downloadCsv(content: string, filename: string) {
  const blob = new Blob([content], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

export default function App() {
  const { data, error, loading, refresh } = useLeaderboard()

  const [selectedViewId, setSelectedViewId] = useState<string | null>(null)
  const [selectedVariant, setSelectedVariant] = useState<string>('')
  const [selectedDataset, setSelectedDataset] = useState<string>('All')
  const [viewOptions, setViewOptions] = useState<TaskViewOptions>({
    showStd: false, sortColumn: null, sortMetric: null, sortDirection: 'asc',
  })

  const activeView = useMemo(() => {
    if (!data?.views.length) return null
    return data.views.find(v => v.id === selectedViewId) ?? data.views[0]
  }, [data, selectedViewId])

  const effectiveVariant = useMemo(() => {
    if (!activeView) return ''
    if (activeView.variants.includes(selectedVariant)) return selectedVariant
    return activeView.variants[0] ?? ''
  }, [activeView, selectedVariant])

  const { rows, columnDefs } = useTaskView(
    activeView ?? null,
    effectiveVariant,
    selectedDataset,
    viewOptions,
  )

  const handleViewSelect = useCallback((id: string) => {
    setSelectedViewId(id)
    setSelectedVariant('')
    setSelectedDataset('All')
  }, [])

  const handleVariantChange = useCallback((v: string) => {
    setSelectedVariant(v)
    setSelectedDataset('All')
  }, [])

  const handleExportCsv = useCallback(() => {
    if (!activeView) return
    downloadCsv(
      rowsToCsv(rows, columnDefs, activeView.primary_metrics),
      `leaderboard_${activeView.id}.csv`,
    )
  }, [rows, columnDefs, activeView])

  if (loading) return (
    <div className="min-h-screen flex items-center justify-center text-gray-500">
      Loading leaderboard data…
    </div>
  )

  if (error) return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-red-600 text-center">
        <p className="font-semibold">Failed to load leaderboard_data.json</p>
        <p className="text-sm mt-1">{error}</p>
        <p className="text-sm mt-2 text-gray-500">
          Run: <code className="bg-gray-100 px-1 rounded">python scripts/build_leaderboard.py</code>
        </p>
      </div>
    </div>
  )

  if (!data || !activeView) return null

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center gap-3">
        <h1 className="text-lg font-bold text-gray-900">torch-timeseries Leaderboard</h1>
        <span className="text-xs text-gray-400">
          Generated {new Date(data.generated_at).toLocaleString()}
        </span>
      </header>

      <ViewSelector
        views={data.views}
        selectedId={activeView.id}
        onSelect={handleViewSelect}
      />

      <VariantBar
        variants={activeView.variants}
        selectedVariant={effectiveVariant}
        viewOptions={viewOptions}
        resultCount={rows.length}
        onVariantChange={handleVariantChange}
        onOptionsChange={setViewOptions}
        onExportCsv={handleExportCsv}
        onRefresh={isLiveServer ? refresh : undefined}
      />

      <div className="flex flex-1 overflow-hidden">
        <DatasetSelector
          datasets={activeView.datasets}
          selected={selectedDataset}
          onChange={setSelectedDataset}
        />

        <main className="flex-1 overflow-auto">
          <TaskTable
            rows={rows}
            columnDefs={columnDefs}
            primaryMetrics={activeView.primary_metrics}
            viewOptions={viewOptions}
            onSortChange={(col, metric, dir) =>
              setViewOptions(o => ({ ...o, sortColumn: col, sortMetric: metric, sortDirection: dir }))
            }
          />
        </main>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Delete old unused files**

```bash
cd /data/yww/notebook/pytorchtimseries/webapp/src
rm components/FilterSidebar.tsx components/ToolBar.tsx
rm hooks/useFiltered.ts hooks/useFiltered.test.ts
```

- [ ] **Step 3: Build the frontend to verify no TypeScript/compile errors**

```bash
cd /data/yww/notebook/pytorchtimseries/webapp
npm run build 2>&1 | tail -20
```

Expected: Build succeeds (exit 0). If there are TypeScript errors in TaskTable or elsewhere, fix them.

- [ ] **Step 4: Run remaining frontend tests**

```bash
npx vitest run
```

Expected: All pass (MetricCell, DatasetSelector, TaskTable, useTaskView tests).

- [ ] **Step 5: Commit**

```bash
cd /data/yww/notebook/pytorchtimseries
git add webapp/src/App.tsx
git rm webapp/src/components/FilterSidebar.tsx webapp/src/components/ToolBar.tsx
git rm webapp/src/hooks/useFiltered.ts webapp/src/hooks/useFiltered.test.ts
git commit -m "feat: rewire App.tsx to task-view layout; remove old FilterSidebar/ToolBar/useFiltered"
```

---

## Task 10: Update CI workflow + README + generate sample data

**Files:**
- Modify: `.github/workflows/leaderboard.yml`
- Modify: `webapp/README.md`

- [ ] **Step 1: Update .github/workflows/leaderboard.yml**

```yaml
name: Deploy Leaderboard

on:
  push:
    branches: [main]
    paths:
      - 'leaderboard_results/**'
      - 'results/**'
      - 'leaderboard/views/**'
      - 'leaderboard/entries/**'
      - 'leaderboard.yaml'
      - 'webapp/**'
      - 'scripts/build_leaderboard.py'

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Python deps
        run: pip install pyyaml

      - name: Build leaderboard data
        run: python scripts/build_leaderboard.py

      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: webapp/package-lock.json

      - name: Install Node deps
        run: cd webapp && npm ci

      - name: Build frontend
        run: cd webapp && npm run build

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./webapp/dist
```

- [ ] **Step 2: Regenerate leaderboard_data.json so the dev server shows the new view format**

```bash
cd /data/yww/notebook/pytorchtimseries
python scripts/build_leaderboard.py
```

Verify `webapp/public/leaderboard_data.json` has a `"views"` key (not `"entries"`).

- [ ] **Step 3: Update webapp/README.md**

```markdown
# torch-timeseries Leaderboard

Task-oriented benchmarking webapp. Views are defined in `leaderboard/views/*.yaml`; results live in `leaderboard_results/` or `results/`.

## Quick start

```bash
# 1. Generate data
python scripts/build_leaderboard.py

# 2. Install frontend deps (first time only)
cd webapp && npm install

# 3. Start dev server
npm run dev
# → http://localhost:5173
```

## Adding benchmark results

Drop a `metrics.json` in `leaderboard_results/{Model}/{Task}/{Dataset}/seed{N}/`:

```json
{
  "model": "DLinear",
  "task": "Forecast",
  "dataset": "ETTh1",
  "seed": 1,
  "hparams": {"windows": 96, "pred_len": 96},
  "metrics": {"mse": 0.384, "mae": 0.276}
}
```

Then re-run `python scripts/build_leaderboard.py`.

## Adding paper baselines

Add a YAML entry in `leaderboard/entries/`:

```yaml
- model: DLinear
  task: Forecast
  dataset: ETTh1
  hparams: {windows: 96, pred_len: 96}
  metrics: {mse: 0.40, mae: 0.30}
  citation: "DLinear (Zeng 2023)"
  url: https://arxiv.org/abs/2205.13504
```

## View configuration

Each task has a view YAML in `leaderboard/views/`. Views are **display-only** — they control how results are grouped and shown, not how experiments are run.

| Task | View file | Sub-columns |
|------|-----------|-------------|
| Long-Term Forecast | `long_term_forecast.yaml` | pred_len (96/192/336/720) |
| Short-Term Forecast | `short_term_forecast.yaml` | pred_len (24/48/96) |
| Imputation | `imputation.yaml` | mask_rate (0.125–0.5) |
| Anomaly Detection | `anomaly_detection.yaml` | none |
| UEA Classification | `uea_classification.yaml` | none |
| Irregular Classification | `irregular_classification.yaml` | none |

## Live local server

```bash
python scripts/build_leaderboard.py
cd webapp && npm run build
python scripts/serve_leaderboard.py
# → http://localhost:8000
```

Click **↺ Refresh** to re-run the build script without restarting.

## GitHub Pages (static deploy)

Push to `main` — CI builds and deploys automatically when `leaderboard_results/`, `leaderboard/views/`, `leaderboard/entries/`, `leaderboard.yaml`, `webapp/`, or `scripts/build_leaderboard.py` changes.
```

- [ ] **Step 4: Run full test suite one final time**

```bash
cd /data/yww/notebook/pytorchtimseries
python -m pytest tests/leaderboard/ -v && cd webapp && npx vitest run
```

Expected: All Python and frontend tests pass.

- [ ] **Step 5: Commit**

```bash
cd /data/yww/notebook/pytorchtimseries
git add .github/workflows/leaderboard.yml webapp/README.md webapp/public/leaderboard_data.json
git commit -m "feat: update CI workflow and README for task-view leaderboard; regenerate data"
```
