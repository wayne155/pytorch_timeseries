# Task-Oriented Leaderboard — Design Spec

**Date:** 2026-06-09

## Goal

Redesign the leaderboard around fixed benchmark tasks (e.g., `LongTermForecast-I96`) rather than a flat list of runs. Each task has predefined sub-columns (pred_lens, anomaly ratios, etc.). The table shows mean across seeds by default, with an optional ±std toggle. Results live in a dedicated `leaderboard_results/` directory separate from ad-hoc `results/`.

---

## Directory Structure

```
leaderboard/
├── views/                          # display-only task view configs (NOT execution configs)
│   ├── long_term_forecast.yaml
│   ├── short_term_forecast.yaml
│   ├── anomaly_detection.yaml
│   ├── imputation.yaml
│   ├── uea_classification.yaml
│   └── irregular_classification.yaml
└── entries/                        # curated paper baselines (existing, unchanged)
    └── *.yaml

leaderboard_results/                # benchmark runs (separate from results/)
    {Model}/
        {Task}/                     # e.g. Forecast, UEAClassification
            {Dataset}/
                seed{N}/
                    metrics.json
                    # model.pt  ← placeholder, not implemented yet

results/                            # ad-hoc experiment runs (unchanged)
```

`leaderboard_results/` is for structured benchmark runs: fixed task, 5 seeds, standard datasets.
`results/` is for ad-hoc runs. The build script reads both (leaderboard_results first).

---

## View YAML Format

View YAMLs are **display-only** — they tell the leaderboard how to group and present results. They do not control how experiments are run.

### Long-Term Forecast

```yaml
# leaderboard/views/long_term_forecast.yaml
id: long_term_forecast
display_name: "Long-Term Forecast"
task: Forecast                        # matches result JSON task field
primary_metrics: [mse, mae]

variants:                             # sub-task groupings (become column groups)
  - label: "I96"                     # LongTermForecast-I96
    match: {windows: 96}
    subcolumns:                       # pred_len → column label
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

### UEA Classification

```yaml
id: uea_classification
display_name: "UEA Classification"
task: UEAClassification
primary_metrics: [accuracy]

variants:
  - label: "W96"
    match: {windows: 96}
    subcolumns: []                    # no sub-columns; single result per dataset

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

### Anomaly Detection

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

---

## `metrics.json` Format (per seed)

```json
{
  "model": "DLinear",
  "task": "Forecast",
  "dataset": "ETTh1",
  "seed": 1,
  "hparams": {
    "windows": 96,
    "pred_len": 96,
    "horizon": 1,
    "batch_size": 32,
    "lr": 0.001
  },
  "metrics": {
    "mse": 0.3843,
    "mae": 0.2756
  },
  "num_params": 4096,
  "train_time_sec": 120.4,
  "git_commit": "abc123",
  "timestamp": "2026-06-09T10:00:00"
}
```

All hparams are stored (no filtering). The view YAML's `match` fields select which results belong to which sub-column.

---

## `leaderboard_data.json` — New Format

The build script aggregates seeds and organizes into a task-view structure.

```jsonc
{
  "generated_at": "2026-06-09T10:00:00",
  "views": [
    {
      "id": "long_term_forecast",
      "display_name": "Long-Term Forecast",
      "primary_metrics": ["mse", "mae"],
      "variants": ["I96", "I336"],
      "datasets": ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Weather", "Traffic"],
      "subcolumns": ["96", "192", "336", "720"],  // pred_len labels for current variant
      "models": [
        {
          "name": "DLinear",
          "source_type": "local_run",
          "citation": "",
          "url": "",
          // results[variant][dataset][subcolumn or "avg"][metric] = {mean, std, n_seeds}
          "results": {
            "I96": {
              "ETTh1": {
                "avg": {
                  "mse": {"mean": 0.417, "std": 0.008, "n_seeds": 5},
                  "mae": {"mean": 0.300, "std": 0.004, "n_seeds": 5}
                },
                "96":  {"mse": {"mean": 0.384, "std": 0.007, "n_seeds": 5}, "mae": {...}},
                "192": {"mse": {"mean": 0.412, "std": 0.008, "n_seeds": 5}, "mae": {...}},
                "336": {"mse": {"mean": 0.423, "std": 0.009, "n_seeds": 5}, "mae": {...}},
                "720": {"mse": {"mean": 0.449, "std": 0.011, "n_seeds": 5}, "mae": {...}}
              }
            }
          }
        }
      ]
    }
  ],
  "schema": {
    "views": ["long_term_forecast", "uea_classification", "anomaly_detection"],
    "datasets_by_view": {
      "long_term_forecast": ["ETTh1", "ETTh2", ...]
    }
  }
}
```

`avg` is computed as the arithmetic mean of the subcolumn means (mean of means across pred_lens).

---

## Build Script: `scripts/build_leaderboard.py`

### Inputs

1. `leaderboard_results/**/{Model}/{Task}/{Dataset}/seed{N}/metrics.json` — benchmark results
2. `results/**` — ad-hoc results (fallback; `leaderboard_results` takes priority for same key)
3. `leaderboard/views/*.yaml` — view configs
4. `leaderboard/entries/*.yaml` — curated paper baselines (existing format, unchanged)

### Algorithm

```
for each view YAML:
    for each variant in view:
        for each dataset:
            collect all matching metrics.json files
            (from leaderboard_results/ first, then results/)
            group by (model, seed) → per-seed values
            group by model → aggregate: mean ± std over seeds
            compute avg subcolumn = mean of subcolumn means
    emit view block
```

### Config: `leaderboard.yaml` (project root)

```yaml
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

---

## Frontend Redesign

### Layout

```
┌───────────────────────────────────────────────────────────────────────────┐
│  torch-timeseries Leaderboard                                              │
├───────────────────────────────────────────────────────────────────────────┤
│  [Long-Term Forecast ▾]  [I96 ▾]          [☐ ±std]  [Export CSV]         │
├──────────────┬────────────────────────────────────────────────────────────┤
│ Datasets     │  Model    │  avg ↓      │   96        │ 192  │ 336  │ 720  │
│              │           │  MSE  MAE   │  MSE  MAE   │ ...  │ ...  │      │
│ ● All        ├───────────┼─────────────┼─────────────┼──────┼──────┼──────┤
│ ○ ETTh1      │ DLinear   │ 0.417 0.300 │ 0.384 0.276 │ ...  │ ...  │ [⋯] │
│ ○ ETTh2      │ iTransf.  │ 0.392 0.281 │ 0.360 0.261 │ ...  │ ...  │ [⋯] │
│ ○ ETTm1      │ PatchTST  │ 0.381 0.274 │ 0.351 0.255 │ ...  │ ...  │ [⋯] │
│ ○ ETTm2      │           │             │             │      │      │      │
│ ○ Electricity│           │             │             │      │      │      │
│ ○ Weather    │           │             │             │      │      │      │
│ ○ Traffic    │           │             │             │      │      │      │
└──────────────┴───────────┴─────────────┴─────────────┴──────┴──────┴──────┘
```

When a specific dataset is selected (e.g., ETTh1), the table columns change:

```
│ Datasets     │  Model    │  avg ↓      │   96        │  192  │  336  │  720  │
│              │           │  MSE  MAE   │  MSE  MAE   │  ...  │  ...  │       │
│ ○ All        ├───────────┼─────────────┼─────────────┼───────┼───────┼───────┤
│ ● ETTh1      │ DLinear   │ 0.417 0.300 │ 0.384 0.276 │  ...  │  ...  │  [⋯] │
│ ○ ETTh2      │ iTransf.  │ 0.392 0.281 │ 0.360 0.261 │  ...  │  ...  │  [⋯] │
```

For tasks without sub-columns (UEA, Anomaly, IrregularClassification), selecting "All" shows one column per dataset + an avg column on the left:

```
│ Datasets       │  Model    │  avg ↑ │ EthanolC │ FaceDetect │ Handwriting │ ...  │
│                │           │  acc   │  acc     │  acc       │  acc        │      │
│ ● All          ├───────────┼────────┼──────────┼────────────┼─────────────┼──────┤
│ ○ EthanolC     │ PatchTST  │  0.78  │  0.71    │  0.75      │  0.45       │ [⋯] │
│ ○ FaceDetect   │ GRU-D     │  0.72  │  0.65    │  0.71      │  0.38       │ [⋯] │
│ ○ Handwriting  │           │        │          │            │             │      │
│ ○ Heartbeat    │           │        │          │            │             │      │
└────────────────┴───────────┴────────┴──────────┴────────────┴─────────────┴──────┘
```

Selecting a specific dataset collapses to a single metric column for that dataset.

Each row has a **`[⋯]` action button** on the far right. What it does is TBD — placeholder for now. Renders as a small `⋯` icon button (`...`), disabled/no-op until behaviour is decided.

**Intended future use (not implemented):** clicking opens a per-seed detail panel:

```
DLinear  ·  ETTh1  ·  LongTermForecast-I96
────────────────────────────────────────────
seed 1   MSE 0.37  MAE 0.27   [↓ model.pt]
seed 2   MSE 0.39  MAE 0.29   [↓ model.pt]
seed 3   MSE 0.38  MAE 0.28   [↓ model.pt]
seed 4   MSE 0.36  MAE 0.26   [↓ model.pt]
seed 5   MSE 0.40  MAE 0.30   [↓ model.pt]
────────────────────────────────────────────
mean     0.38 ± 0.01
```

With ±std enabled:
```
│ DLinear   │ 0.417±0.008 │ 0.384±0.007 │ ...  │ [⋯] │
```

### Controls (ToolBar, top row)

| Control | Behavior |
|---------|----------|
| View selector | Switches between task views (LongTermForecast, UEAClassification, …) |
| Variant selector | Switches between I96 / I336 / … within a view |
| Dataset selector | Left-side panel; radio buttons: "All" + one per dataset in the current view YAML. Selecting "All" shows all datasets. Selecting a specific dataset filters the table to that dataset's data only (see layout mockups above). |
| ±std checkbox | Toggle std display in cells |
| Export CSV | Downloads current visible table |

**Dataset selector behavior by task type:**

- **Tasks with subcolumns (Forecast, Imputation):** Selecting "All" shows avg across all datasets. Selecting ETTh1 shows only ETTh1's avg + subcolumn columns (pred_len or mask_rate).
- **Tasks without subcolumns (UEA, Anomaly, IrregularClassification):** Selecting "All" shows one column per dataset + a left-side avg column. Selecting a specific dataset collapses to a single metric column for that dataset.

The list of available datasets is read from the active view YAML's `datasets` field. Datasets with no results in the current data are shown but disabled.

### Components

| Component | Role |
|-----------|------|
| `src/types.ts` | New types: `ViewEntry`, `ModelResult`, `SubcolumnResult` |
| `src/hooks/useTaskView.ts` | Selects and filters the active view/variant/dataset from data |
| `src/components/TaskTable.tsx` | Multi-level header table (avg + subcolumns) using TanStack Table column groups |
| `src/components/ViewSelector.tsx` | Top-level tab/dropdown for view selection |
| `src/components/DatasetSelector.tsx` | Left-side panel with radio buttons (All + one per dataset); grays out datasets with no data; emits selected dataset to parent |
| `src/components/VariantBar.tsx` | Variant selector + std toggle + export (dataset selector moved to its own component) |
| `src/components/MetricCell.tsx` | Reuse existing (mean, optional ±std, best/worst highlight) |
| `src/components/RowActionButton.tsx` | `[⋯]` placeholder button, far-right column; no-op for now |

### Sorting

Any metric header is clickable to sort the table by that metric.

- First click: ascending for lower-is-better metrics (mse, mae, loss, …), descending for higher-is-better (accuracy, f1, …)
- Second click on same header: reverses direction
- Active sort column shows ▲ or ▼ in header
- Default sort on load: avg column, primary metric (first in `primary_metrics`), lower-is-better direction

Sort key is a tuple `(subcolumn, metric)` — e.g., clicking the MAE header under the "192" subcolumn sorts by `("192", "mae")`. Clicking the MAE header under "avg" sorts by `("avg", "mae")`.

When dataset selector is "All", models may have results for multiple datasets; in that case sort uses the mean of the selected metric across all datasets.

### Best/Worst Highlight

Per metric column, across all models: best = green, worst = faint red (same as current). Highlight respects the same lower/higher-is-better direction as sorting.

---

## Testing

**Python (`tests/leaderboard/test_build_task_view.py`)**:
- `test_ingest_metrics_json` — reads a seed metrics.json, returns correct entry
- `test_aggregate_seeds` — 5 seeds → correct mean ± std
- `test_avg_subcolumn` — avg = mean of [p96, p192, p336, p720] means
- `test_local_priority` — same key in both dirs → leaderboard_results wins
- `test_view_yaml_loads` — all view YAMLs parse without error
- `test_build_end_to_end` — fixture data → correct leaderboard_data.json structure

**Frontend (Vitest)**:
- `useTaskView.test.ts` — variant switch changes visible subcolumns
- `useTaskView.test.ts` — dataset filter isolates one dataset's results
- `TaskTable.test.tsx` — renders avg + 4 subcolumn headers
- `MetricCell.test.tsx` — existing tests still pass (component unchanged)

---

## What Is NOT in This Spec

- Remote backend (placeholder in `leaderboard.yaml`, not implemented)
- Model checkpoint storage (future)
- `[⋯]` row action button behaviour (placeholder added, TBD)
- Seed detail panel / checkpoint download (intended future use of `[⋯]`)
- Cross-task averaging (dropped)
- Running benchmark script (`scripts/run_benchmark.py`) — separate spec
