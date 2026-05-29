# Leaderboard Webapp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deployable React leaderboard webpage that reads `results/*.json` + curated YAML entries, lets users filter by task/dataset/model/hparams, and toggles between per-seed rows and aggregated mean±std.

**Architecture:** A Python build script (`scripts/build_leaderboard.py`) ingests all result files and writes `webapp/public/leaderboard_data.json`; the React+Vite frontend loads this JSON at runtime and does all filtering/aggregation client-side. For live local use, `scripts/serve_leaderboard.py` (FastAPI) serves the built frontend and exposes `/api/refresh` to rebuild the data without restarting.

**Tech Stack:** Python 3.10+, PyYAML, FastAPI, uvicorn; Node.js 20, React 18, Vite 5, TanStack Table v8, Tailwind CSS 3, TypeScript 5, Vitest 1.

---

## File Map

```
webapp/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.js
├── postcss.config.js
├── index.html
├── src/
│   ├── main.tsx
│   ├── index.css
│   ├── test-setup.ts
│   ├── types.ts
│   ├── App.tsx
│   ├── hooks/
│   │   ├── useLeaderboard.ts
│   │   ├── useFiltered.ts
│   │   └── useFiltered.test.ts
│   └── components/
│       ├── FilterSidebar.tsx
│       ├── LeaderboardTable.tsx
│       ├── MetricCell.tsx
│       ├── MetricCell.test.tsx
│       ├── HparamBadges.tsx
│       └── ToolBar.tsx
└── public/
    └── leaderboard_data.json   (generated — gitignored for dev, committed for gh-pages)

scripts/
├── build_leaderboard.py
└── serve_leaderboard.py

tests/
└── leaderboard/
    └── test_build_script.py

.github/
└── workflows/
    └── leaderboard.yml
```

---

## Task 1: Node.js + webapp scaffold

**Files:**
- Create: `webapp/package.json`
- Create: `webapp/vite.config.ts`
- Create: `webapp/tsconfig.json`
- Create: `webapp/tailwind.config.js`
- Create: `webapp/postcss.config.js`
- Create: `webapp/index.html`
- Create: `webapp/src/index.css`
- Create: `webapp/src/test-setup.ts`
- Create: `webapp/src/main.tsx`
- Create: `webapp/src/App.tsx`

- [ ] **Step 1: Install Node.js 20 and npm (if not present)**

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc   # or ~/.zshrc
nvm install 20
nvm use 20
node --version   # expected: v20.x.x
npm --version    # expected: 10.x.x
```

- [ ] **Step 2: Write `webapp/package.json`**

```json
{
  "name": "leaderboard",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest"
  },
  "dependencies": {
    "@tanstack/react-table": "^8.10.7",
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.4.2",
    "@testing-library/react": "^15.0.7",
    "@testing-library/user-event": "^14.5.2",
    "@types/react": "^18.3.1",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.0",
    "autoprefixer": "^10.4.19",
    "jsdom": "^24.0.0",
    "postcss": "^8.4.38",
    "tailwindcss": "^3.4.3",
    "typescript": "^5.4.5",
    "vite": "^5.2.11",
    "vitest": "^1.6.0"
  }
}
```

- [ ] **Step 3: Write config files**

`webapp/vite.config.ts`:
```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: './',
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test-setup.ts',
  },
})
```

`webapp/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"]
}
```

`webapp/tailwind.config.js`:
```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: { extend: {} },
  plugins: [],
}
```

`webapp/postcss.config.js`:
```js
export default { plugins: { tailwindcss: {}, autoprefixer: {} } }
```

- [ ] **Step 4: Write `webapp/index.html`, CSS, and entry point**

`webapp/index.html`:
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>torch-timeseries Leaderboard</title>
  </head>
  <body class="bg-gray-50">
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

`webapp/src/index.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

`webapp/src/test-setup.ts`:
```ts
import '@testing-library/jest-dom'
```

`webapp/src/main.tsx`:
```tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><App /></React.StrictMode>
)
```

`webapp/src/App.tsx` (skeleton — will be replaced in Task 9):
```tsx
export default function App() {
  return <div className="min-h-screen p-8 text-gray-700">Loading...</div>
}
```

- [ ] **Step 5: Install deps and verify dev server starts**

```bash
cd webapp
npm install
npm run dev
```

Expected: Vite dev server starts on http://localhost:5173, browser shows "Loading..."

- [ ] **Step 6: Verify tests run**

```bash
cd webapp
npm test -- --run
```

Expected: "No test files found" (no tests yet) and exit 0.

- [ ] **Step 7: Commit**

```bash
git add webapp/
git commit -m "feat: scaffold leaderboard webapp (Vite + React + Tailwind)"
```

---

## Task 2: TypeScript types (`src/types.ts`)

**Files:**
- Create: `webapp/src/types.ts`

- [ ] **Step 1: Write `webapp/src/types.ts`**

```ts
export interface Entry {
  id: string
  model: string
  task: string
  dataset: string
  seed: number | null
  hparams: Record<string, number | string | boolean>
  metrics: Record<string, number>
  num_params: number | null
  train_time_sec: number | null
  git_commit: string
  timestamp: string
  source_type: 'local_run' | 'paper'
  citation: string
  url: string
  notes: string
}

export interface AggregatedMetric {
  mean: number
  std: number
}

/** One row in the table — either a per-seed Entry or an aggregated group. */
export interface DisplayRow {
  key: string
  model: string
  task: string
  dataset: string
  hparams: Record<string, number | string | boolean>
  metrics: Record<string, number | AggregatedMetric>
  num_seeds: number
  seed: number | null
  source_type: string
  citation: string
  url: string
  isAggregated: boolean
}

export interface LeaderboardSchema {
  tasks: string[]
  datasets_by_task: Record<string, string[]>
  models: string[]
  hparams_by_task: Record<string, string[]>
  hparam_options: Record<string, Record<string, (number | string)[]>>
}

export interface LeaderboardData {
  generated_at: string
  entries: Entry[]
  schema: LeaderboardSchema
}

export interface Filters {
  task: string | null
  datasets: string[]
  models: string[]
  hparams: Record<string, string | null>
}

export interface ViewOptions {
  aggregate: boolean
  showStd: boolean
  visibleMetrics: string[]
  sortColumn: string | null
  sortDirection: 'asc' | 'desc'
}
```

- [ ] **Step 2: Verify TypeScript compiles cleanly**

```bash
cd webapp
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add webapp/src/types.ts
git commit -m "feat: add TypeScript types for leaderboard"
```

---

## Task 3: Python build script (TDD)

**Files:**
- Create: `scripts/build_leaderboard.py`
- Create: `tests/leaderboard/__init__.py`
- Create: `tests/leaderboard/test_build_script.py`

- [ ] **Step 1: Install Python dependencies**

```bash
pip install pyyaml fastapi uvicorn
```

Expected: installs without error. Verify: `python3 -c "import yaml, fastapi; print('ok')"` → `ok`.

- [ ] **Step 2: Write the tests first**

`tests/leaderboard/__init__.py`: (empty)

`tests/leaderboard/test_build_script.py`:
```python
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
```

- [ ] **Step 3: Run tests, verify they fail**

```bash
cd /data/yww/notebook/pytorchtimseries
python -m pytest tests/leaderboard/ -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'scripts.build_leaderboard'`

- [ ] **Step 4: Create `scripts/__init__.py` and write `scripts/build_leaderboard.py`**

`scripts/__init__.py`: (empty)

`scripts/build_leaderboard.py`:
```python
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
```

- [ ] **Step 5: Run tests, verify they pass**

```bash
python -m pytest tests/leaderboard/ -v
```

Expected: `6 passed`

- [ ] **Step 6: Smoke-run the script against real results**

```bash
python scripts/build_leaderboard.py
```

Expected: `Wrote N entries → webapp/public/leaderboard_data.json`

```bash
python -c "import json; d=json.load(open('webapp/public/leaderboard_data.json')); print(len(d['entries']), 'entries,', d['schema']['tasks'])"
```

Expected: prints entry count and task list without error.

- [ ] **Step 7: Commit**

```bash
git add scripts/ tests/leaderboard/ webapp/public/leaderboard_data.json
git commit -m "feat: add build_leaderboard.py with pytest tests"
```

---

## Task 4: `useLeaderboard` + `useFiltered` hooks (TDD)

**Files:**
- Create: `webapp/src/hooks/useLeaderboard.ts`
- Create: `webapp/src/hooks/useFiltered.ts`
- Create: `webapp/src/hooks/useFiltered.test.ts`

- [ ] **Step 1: Write `useFiltered.test.ts` first**

`webapp/src/hooks/useFiltered.test.ts`:
```ts
import { describe, it, expect } from 'vitest'
import { renderHook } from '@testing-library/react'
import { useFiltered, isLowerBetter, getBestWorst } from './useFiltered'
import type { Entry, Filters, ViewOptions } from '../types'

function makeEntry(overrides: Partial<Entry> = {}): Entry {
  return {
    id: 'id1', model: 'PatchTST', task: 'Forecast', dataset: 'ETTh1',
    seed: 1, hparams: { windows: 96, pred_len: 96, horizon: 1 },
    metrics: { mse: 0.38, mae: 0.28 }, num_params: null, train_time_sec: null,
    git_commit: '', timestamp: '', source_type: 'local_run',
    citation: '', url: '', notes: '', ...overrides,
  }
}

const VIEW_IND: ViewOptions = {
  aggregate: false, showStd: false,
  visibleMetrics: ['mse', 'mae'], sortColumn: null, sortDirection: 'asc',
}
const VIEW_AGG: ViewOptions = { ...VIEW_IND, aggregate: true }
const NO_FILTER: Filters = { task: null, datasets: [], models: [], hparams: {} }

describe('isLowerBetter', () => {
  it('returns true for mse', () => expect(isLowerBetter('mse')).toBe(true))
  it('returns true for cross_entropy_loss', () => expect(isLowerBetter('cross_entropy_loss')).toBe(true))
  it('returns false for accuracy', () => expect(isLowerBetter('accuracy')).toBe(false))
})

describe('useFiltered – individual mode', () => {
  it('returns all rows when no filters active', () => {
    const entries = [makeEntry({ id: 'a' }), makeEntry({ id: 'b' })]
    const { result } = renderHook(() => useFiltered(entries, NO_FILTER, VIEW_IND))
    expect(result.current).toHaveLength(2)
    expect(result.current[0].isAggregated).toBe(false)
  })

  it('filters by task', () => {
    const entries = [
      makeEntry({ id: 'a', task: 'Forecast' }),
      makeEntry({ id: 'b', task: 'UEAClassification' }),
    ]
    const filters: Filters = { ...NO_FILTER, task: 'Forecast' }
    const { result } = renderHook(() => useFiltered(entries, filters, VIEW_IND))
    expect(result.current).toHaveLength(1)
    expect(result.current[0].task).toBe('Forecast')
  })

  it('filters by dataset (multi-select)', () => {
    const entries = [
      makeEntry({ id: 'a', dataset: 'ETTh1' }),
      makeEntry({ id: 'b', dataset: 'ETTm1' }),
    ]
    const filters: Filters = { ...NO_FILTER, datasets: ['ETTh1'] }
    const { result } = renderHook(() => useFiltered(entries, filters, VIEW_IND))
    expect(result.current).toHaveLength(1)
  })

  it('filters by hparam value', () => {
    const entries = [
      makeEntry({ id: 'a', hparams: { windows: 96, pred_len: 96, horizon: 1 } }),
      makeEntry({ id: 'b', hparams: { windows: 336, pred_len: 96, horizon: 1 } }),
    ]
    const filters: Filters = { ...NO_FILTER, hparams: { windows: '96' } }
    const { result } = renderHook(() => useFiltered(entries, filters, VIEW_IND))
    expect(result.current).toHaveLength(1)
    expect(result.current[0].hparams.windows).toBe(96)
  })
})

describe('useFiltered – aggregate mode', () => {
  it('groups two seeds into one row', () => {
    const entries = [
      makeEntry({ id: 'a', seed: 1, metrics: { mse: 0.40 } }),
      makeEntry({ id: 'b', seed: 2, metrics: { mse: 0.36 } }),
    ]
    const { result } = renderHook(() => useFiltered(entries, NO_FILTER, VIEW_AGG))
    expect(result.current).toHaveLength(1)
    expect(result.current[0].isAggregated).toBe(true)
    expect(result.current[0].num_seeds).toBe(2)
    const mse = result.current[0].metrics['mse'] as { mean: number; std: number }
    expect(mse.mean).toBeCloseTo(0.38, 5)
  })

  it('keeps different hparam configs as separate rows', () => {
    const entries = [
      makeEntry({ id: 'a', hparams: { windows: 96, pred_len: 96, horizon: 1 } }),
      makeEntry({ id: 'b', hparams: { windows: 336, pred_len: 96, horizon: 1 } }),
    ]
    const { result } = renderHook(() => useFiltered(entries, NO_FILTER, VIEW_AGG))
    expect(result.current).toHaveLength(2)
  })
})

describe('getBestWorst', () => {
  it('returns min value as best for lower-is-better metrics', () => {
    const rows = [
      { ...makeEntry({ metrics: { mse: 0.4 } }), key: 'a', isAggregated: false, num_seeds: 1 },
      { ...makeEntry({ metrics: { mse: 0.3 } }), key: 'b', isAggregated: false, num_seeds: 1 },
    ]
    const { best, worst } = getBestWorst(rows as never, 'mse')
    expect(best).toBe(0.3)
    expect(worst).toBe(0.4)
  })
})
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
cd webapp && npm test -- --run
```

Expected: fails with `Cannot find module './useFiltered'`

- [ ] **Step 3: Write `useLeaderboard.ts`**

`webapp/src/hooks/useLeaderboard.ts`:
```ts
import { useState, useEffect } from 'react'
import type { LeaderboardData } from '../types'

export function useLeaderboard(url = './leaderboard_data.json') {
  const [data, setData] = useState<LeaderboardData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const load = () => {
    setLoading(true)
    setError(null)
    fetch(url)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json() as Promise<LeaderboardData>
      })
      .then(d => { setData(d); setLoading(false) })
      .catch((e: Error) => { setError(e.message); setLoading(false) })
  }

  useEffect(() => { load() }, [url])
  return { data, error, loading, refresh: load }
}
```

- [ ] **Step 4: Write `useFiltered.ts`**

`webapp/src/hooks/useFiltered.ts`:
```ts
import { useMemo } from 'react'
import type { Entry, DisplayRow, Filters, ViewOptions, AggregatedMetric } from '../types'

const LOWER_BETTER_PATTERNS = ['mse', 'mae', 'loss', 'error', 'rmse', 'crps', 'wis', 'nll']

export function isLowerBetter(key: string): boolean {
  const k = key.toLowerCase()
  return LOWER_BETTER_PATTERNS.some(p => k.includes(p))
}

function sortedHparamKey(hparams: Record<string, number | string | boolean>): string {
  return JSON.stringify(Object.fromEntries(Object.entries(hparams).sort()))
}

function groupKey(e: Entry): string {
  return `${e.model}|${e.task}|${e.dataset}|${sortedHparamKey(e.hparams)}`
}

function aggregateGroup(group: Entry[]): DisplayRow {
  const first = group[0]
  const metricKeys = [...new Set(group.flatMap(e => Object.keys(e.metrics)))]
  const metrics: Record<string, AggregatedMetric> = {}
  for (const k of metricKeys) {
    const vals = group.map(e => e.metrics[k]).filter((v): v is number => v !== undefined)
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length
    const variance = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(vals.length - 1, 1)
    metrics[k] = { mean, std: Math.sqrt(variance) }
  }
  return {
    key: groupKey(first),
    model: first.model, task: first.task, dataset: first.dataset,
    hparams: first.hparams, metrics, num_seeds: group.length,
    seed: null, source_type: first.source_type,
    citation: first.citation, url: first.url, isAggregated: true,
  }
}

function entryToRow(e: Entry): DisplayRow {
  return {
    key: e.id, model: e.model, task: e.task, dataset: e.dataset,
    hparams: e.hparams, metrics: e.metrics, num_seeds: 1,
    seed: e.seed, source_type: e.source_type,
    citation: e.citation, url: e.url, isAggregated: false,
  }
}

function applyFilters(entries: Entry[], filters: Filters): Entry[] {
  return entries.filter(e => {
    if (filters.task && e.task !== filters.task) return false
    if (filters.datasets.length && !filters.datasets.includes(e.dataset)) return false
    if (filters.models.length && !filters.models.includes(e.model)) return false
    for (const [key, val] of Object.entries(filters.hparams)) {
      if (val === null) continue
      if (String(e.hparams[key]) !== val) return false
    }
    return true
  })
}

function metricMean(row: DisplayRow, col: string): number | null {
  const m = row.metrics[col]
  if (m === undefined) return null
  return typeof m === 'number' ? m : m.mean
}

export function getBestWorst(rows: DisplayRow[], metricKey: string): { best: number; worst: number } {
  const vals = rows.map(r => metricMean(r, metricKey)).filter((v): v is number => v !== null)
  if (vals.length === 0) return { best: NaN, worst: NaN }
  const asc = isLowerBetter(metricKey)
  return {
    best: asc ? Math.min(...vals) : Math.max(...vals),
    worst: asc ? Math.max(...vals) : Math.min(...vals),
  }
}

export function useFiltered(
  entries: Entry[],
  filters: Filters,
  viewOptions: ViewOptions,
): DisplayRow[] {
  return useMemo(() => {
    const filtered = applyFilters(entries, filters)

    let rows: DisplayRow[]
    if (viewOptions.aggregate) {
      const groups = new Map<string, Entry[]>()
      for (const e of filtered) {
        const k = groupKey(e)
        if (!groups.has(k)) groups.set(k, [])
        groups.get(k)!.push(e)
      }
      rows = Array.from(groups.values()).map(aggregateGroup)
    } else {
      rows = filtered.map(entryToRow)
    }

    if (!viewOptions.sortColumn) return rows
    const col = viewOptions.sortColumn
    const dir = viewOptions.sortDirection === 'asc' ? 1 : -1
    return [...rows].sort((a, b) => {
      const av = col === 'model' ? a.model : (metricMean(a, col) ?? 0)
      const bv = col === 'model' ? b.model : (metricMean(b, col) ?? 0)
      if (typeof av === 'string') return dir * av.localeCompare(bv as string)
      return dir * ((av as number) - (bv as number))
    })
  }, [entries, filters, viewOptions])
}
```

- [ ] **Step 5: Run tests, verify they pass**

```bash
cd webapp && npm test -- --run
```

Expected: `10 passed`

- [ ] **Step 6: Commit**

```bash
git add webapp/src/hooks/
git commit -m "feat: add useLeaderboard and useFiltered hooks"
```

---

## Task 5: `MetricCell` + `HparamBadges` (TDD)

**Files:**
- Create: `webapp/src/components/MetricCell.tsx`
- Create: `webapp/src/components/MetricCell.test.tsx`
- Create: `webapp/src/components/HparamBadges.tsx`

- [ ] **Step 1: Write `MetricCell.test.tsx` first**

`webapp/src/components/MetricCell.test.tsx`:
```tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MetricCell } from './MetricCell'

describe('MetricCell', () => {
  it('renders a plain number value', () => {
    render(<MetricCell value={0.3845} isBest={false} isWorst={false} showStd={false} />)
    expect(screen.getByText('0.3845')).toBeInTheDocument()
  })

  it('renders mean for aggregated metric', () => {
    render(<MetricCell value={{ mean: 0.38, std: 0.01 }} isBest={false} isWorst={false} showStd={false} />)
    expect(screen.getByText('0.3800')).toBeInTheDocument()
  })

  it('shows std when showStd is true and value is aggregated', () => {
    render(<MetricCell value={{ mean: 0.38, std: 0.01 }} isBest={false} isWorst={false} showStd={true} />)
    expect(screen.getByText(/±0.0100/)).toBeInTheDocument()
  })

  it('does not show std when showStd is false', () => {
    render(<MetricCell value={{ mean: 0.38, std: 0.01 }} isBest={false} isWorst={false} showStd={false} />)
    expect(screen.queryByText(/±/)).toBeNull()
  })

  it('applies green class when isBest', () => {
    const { container } = render(
      <MetricCell value={0.3} isBest={true} isWorst={false} showStd={false} />
    )
    expect(container.firstChild).toHaveClass('bg-green-50')
  })

  it('applies red class when isWorst', () => {
    const { container } = render(
      <MetricCell value={0.9} isBest={false} isWorst={true} showStd={false} />
    )
    expect(container.firstChild).toHaveClass('bg-red-50')
  })
})
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
cd webapp && npm test -- --run 2>&1 | grep -E "FAIL|Cannot find"
```

Expected: `Cannot find module './MetricCell'`

- [ ] **Step 3: Write `MetricCell.tsx`**

`webapp/src/components/MetricCell.tsx`:
```tsx
import type { AggregatedMetric } from '../types'

interface MetricCellProps {
  value: number | AggregatedMetric
  isBest: boolean
  isWorst: boolean
  showStd: boolean
}

export function MetricCell({ value, isBest, isWorst, showStd }: MetricCellProps) {
  const numVal = typeof value === 'number' ? value : value.mean
  const std = typeof value === 'object' ? value.std : null

  const cls = isBest
    ? 'bg-green-50 text-green-800 font-semibold'
    : isWorst
    ? 'bg-red-50 text-red-400'
    : ''

  return (
    <span className={`tabular-nums px-1 rounded ${cls}`}>
      {numVal.toFixed(4)}
      {showStd && std !== null && (
        <span className="text-gray-400 text-xs"> ±{std.toFixed(4)}</span>
      )}
    </span>
  )
}
```

- [ ] **Step 4: Write `HparamBadges.tsx`**

`webapp/src/components/HparamBadges.tsx`:
```tsx
const ABBREVS: Record<string, string> = {
  windows: 'w', pred_len: 'p', horizon: 'h',
  mask_rate: 'm', seq_len: 's',
}

interface HparamBadgesProps {
  hparams: Record<string, number | string | boolean>
  onBadgeClick?: (key: string, value: string) => void
}

export function HparamBadges({ hparams, onBadgeClick }: HparamBadgesProps) {
  const pairs = Object.entries(hparams)
  if (pairs.length === 0) return null
  return (
    <div className="flex gap-1 flex-wrap">
      {pairs.map(([k, v]) => (
        <span
          key={k}
          title={`${k}=${v}`}
          className="px-1.5 py-0.5 bg-gray-100 text-gray-600 text-xs rounded cursor-pointer hover:bg-blue-100 select-none"
          onClick={() => onBadgeClick?.(k, String(v))}
        >
          {ABBREVS[k] ?? k[0]}{v}
        </span>
      ))}
    </div>
  )
}
```

- [ ] **Step 5: Run all tests**

```bash
cd webapp && npm test -- --run
```

Expected: all tests pass (MetricCell: 6, useFiltered: 10, total ≥ 16 passed)

- [ ] **Step 6: Commit**

```bash
git add webapp/src/components/MetricCell.tsx webapp/src/components/MetricCell.test.tsx webapp/src/components/HparamBadges.tsx
git commit -m "feat: add MetricCell and HparamBadges components"
```

---

## Task 6: `FilterSidebar`

**Files:**
- Create: `webapp/src/components/FilterSidebar.tsx`

- [ ] **Step 1: Write `FilterSidebar.tsx`**

`webapp/src/components/FilterSidebar.tsx`:
```tsx
import { useState } from 'react'
import type { Filters, LeaderboardSchema } from '../types'

interface FilterSidebarProps {
  schema: LeaderboardSchema
  filters: Filters
  onChange: (f: Filters) => void
}

function SearchCheckboxList({
  label, options, selected, onToggle,
}: { label: string; options: string[]; selected: string[]; onToggle: (v: string) => void }) {
  const [q, setQ] = useState('')
  const visible = q ? options.filter(o => o.toLowerCase().includes(q.toLowerCase())) : options
  return (
    <div className="mb-4">
      <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">{label}</div>
      <input
        className="w-full text-xs border border-gray-200 rounded px-2 py-1 mb-1 focus:outline-none"
        placeholder="Search…"
        value={q}
        onChange={e => setQ(e.target.value)}
      />
      <div className="max-h-40 overflow-y-auto space-y-0.5">
        {visible.map(opt => (
          <label key={opt} className="flex items-center gap-1.5 cursor-pointer text-sm hover:bg-gray-50 px-1 rounded">
            <input
              type="checkbox"
              checked={selected.includes(opt)}
              onChange={() => onToggle(opt)}
              className="rounded"
            />
            <span className="truncate">{opt}</span>
          </label>
        ))}
        {visible.length === 0 && <div className="text-xs text-gray-400 px-1">No matches</div>}
      </div>
    </div>
  )
}

export function FilterSidebar({ schema, filters, onChange }: FilterSidebarProps) {
  const toggleList = (key: keyof Pick<Filters, 'datasets' | 'models'>, val: string) => {
    const cur = filters[key]
    onChange({
      ...filters,
      [key]: cur.includes(val) ? cur.filter(v => v !== val) : [...cur, val],
    })
  }

  const setTask = (task: string) => {
    onChange({ task, datasets: [], models: [], hparams: {} })
  }

  const setHparam = (key: string, val: string) => {
    onChange({ ...filters, hparams: { ...filters.hparams, [key]: val === '' ? null : val } })
  }

  const reset = () => onChange({ task: null, datasets: [], models: [], hparams: {} })

  const currentDatasets = filters.task ? (schema.datasets_by_task[filters.task] ?? []) : []
  const currentHparamKeys = filters.task ? (schema.hparams_by_task[filters.task] ?? []) : []

  return (
    <aside className="w-56 shrink-0 bg-white border-r border-gray-200 p-4 overflow-y-auto">
      <div className="flex items-center justify-between mb-4">
        <span className="font-semibold text-gray-800">Filters</span>
        <button onClick={reset} className="text-xs text-blue-500 hover:underline">Reset all</button>
      </div>

      {/* Task */}
      <div className="mb-4">
        <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Task</div>
        <div className="space-y-0.5">
          <label className="flex items-center gap-1.5 text-sm cursor-pointer hover:bg-gray-50 px-1 rounded">
            <input type="radio" name="task" checked={filters.task === null} onChange={() => reset()} />
            <span>All</span>
          </label>
          {schema.tasks.map(t => (
            <label key={t} className="flex items-center gap-1.5 text-sm cursor-pointer hover:bg-gray-50 px-1 rounded">
              <input type="radio" name="task" checked={filters.task === t} onChange={() => setTask(t)} />
              <span>{t}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Dataset */}
      {currentDatasets.length > 0 && (
        <SearchCheckboxList
          label="Dataset"
          options={currentDatasets}
          selected={filters.datasets}
          onToggle={v => toggleList('datasets', v)}
        />
      )}

      {/* Model */}
      <SearchCheckboxList
        label="Model"
        options={schema.models}
        selected={filters.models}
        onToggle={v => toggleList('models', v)}
      />

      {/* Hparams */}
      {currentHparamKeys.length > 0 && (
        <div className="mb-4">
          <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Hparams</div>
          {currentHparamKeys.map(key => {
            const opts = schema.hparam_options[filters.task!]?.[key] ?? []
            return (
              <div key={key} className="mb-2">
                <div className="text-xs text-gray-500 mb-0.5">{key}</div>
                <select
                  className="w-full text-sm border border-gray-200 rounded px-1.5 py-1 focus:outline-none"
                  value={filters.hparams[key] ?? ''}
                  onChange={e => setHparam(key, e.target.value)}
                >
                  <option value="">All</option>
                  {opts.map(v => (
                    <option key={String(v)} value={String(v)}>{String(v)}</option>
                  ))}
                </select>
              </div>
            )
          })}
        </div>
      )}
    </aside>
  )
}
```

- [ ] **Step 2: Run all tests (no regressions)**

```bash
cd webapp && npm test -- --run
```

Expected: all existing tests still pass.

- [ ] **Step 3: Commit**

```bash
git add webapp/src/components/FilterSidebar.tsx
git commit -m "feat: add FilterSidebar component"
```

---

## Task 7: `ToolBar`

**Files:**
- Create: `webapp/src/components/ToolBar.tsx`

- [ ] **Step 1: Write `ToolBar.tsx`**

`webapp/src/components/ToolBar.tsx`:
```tsx
import type { ViewOptions } from '../types'

interface ToolBarProps {
  viewOptions: ViewOptions
  allMetrics: string[]
  resultCount: number
  onViewChange: (v: ViewOptions) => void
  onExportCsv: () => void
  onRefresh?: () => void
}

export function ToolBar({
  viewOptions, allMetrics, resultCount, onViewChange, onExportCsv, onRefresh,
}: ToolBarProps) {
  const toggle = (patch: Partial<ViewOptions>) => onViewChange({ ...viewOptions, ...patch })

  const toggleMetric = (key: string) => {
    const cur = viewOptions.visibleMetrics
    toggle({
      visibleMetrics: cur.includes(key) ? cur.filter(k => k !== key) : [...cur, key],
    })
  }

  return (
    <div className="flex items-center gap-3 px-4 py-2 border-b border-gray-200 bg-white flex-wrap">
      {/* Aggregate toggle */}
      <button
        className={`text-sm px-3 py-1 rounded border ${viewOptions.aggregate ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700 border-gray-300 hover:border-gray-400'}`}
        onClick={() => toggle({ aggregate: !viewOptions.aggregate })}
      >
        {viewOptions.aggregate ? 'Aggregated' : 'Per seed'}
      </button>

      {/* Show std */}
      {viewOptions.aggregate && (
        <label className="flex items-center gap-1.5 text-sm text-gray-600 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={viewOptions.showStd}
            onChange={e => toggle({ showStd: e.target.checked })}
          />
          Show ±std
        </label>
      )}

      {/* Column visibility */}
      <div className="relative group">
        <button className="text-sm px-3 py-1 rounded border border-gray-300 hover:border-gray-400 bg-white text-gray-700">
          Columns ▾
        </button>
        <div className="absolute left-0 top-full mt-1 bg-white border border-gray-200 rounded shadow-lg p-2 z-10 hidden group-hover:block min-w-max">
          {allMetrics.map(k => (
            <label key={k} className="flex items-center gap-1.5 text-sm cursor-pointer hover:bg-gray-50 px-2 py-0.5 rounded">
              <input
                type="checkbox"
                checked={viewOptions.visibleMetrics.includes(k)}
                onChange={() => toggleMetric(k)}
              />
              {k}
            </label>
          ))}
        </div>
      </div>

      {/* Result count */}
      <span className="text-sm text-gray-500 ml-auto">{resultCount} results</span>

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

- [ ] **Step 2: Run all tests**

```bash
cd webapp && npm test -- --run
```

Expected: all existing tests pass.

- [ ] **Step 3: Commit**

```bash
git add webapp/src/components/ToolBar.tsx
git commit -m "feat: add ToolBar component"
```

---

## Task 8: `LeaderboardTable`

**Files:**
- Create: `webapp/src/components/LeaderboardTable.tsx`

- [ ] **Step 1: Write `LeaderboardTable.tsx`**

`webapp/src/components/LeaderboardTable.tsx`:
```tsx
import { useMemo, useState } from 'react'
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table'
import { MetricCell } from './MetricCell'
import { HparamBadges } from './HparamBadges'
import { getBestWorst, isLowerBetter } from '../hooks/useFiltered'
import type { DisplayRow, ViewOptions } from '../types'

interface LeaderboardTableProps {
  rows: DisplayRow[]
  viewOptions: ViewOptions
  onSortChange: (col: string, dir: 'asc' | 'desc') => void
  onHparamClick: (key: string, value: string) => void
}

export function LeaderboardTable({
  rows, viewOptions, onSortChange, onHparamClick,
}: LeaderboardTableProps) {
  const [sorting, setSorting] = useState<SortingState>([])

  const visibleMetrics = viewOptions.visibleMetrics

  const bestWorstByMetric = useMemo(() => {
    const result: Record<string, { best: number; worst: number }> = {}
    for (const key of visibleMetrics) {
      result[key] = getBestWorst(rows, key)
    }
    return result
  }, [rows, visibleMetrics])

  const columns = useMemo<ColumnDef<DisplayRow>[]>(() => {
    const staticCols: ColumnDef<DisplayRow>[] = [
      {
        id: 'rank',
        header: '#',
        cell: info => info.row.index + 1,
        enableSorting: false,
        size: 48,
      },
      {
        accessorKey: 'model',
        header: 'Model',
        cell: info => {
          const row = info.row.original
          return row.url
            ? <a href={row.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">{row.model}</a>
            : row.model
        },
      },
      {
        accessorKey: 'dataset',
        header: 'Dataset',
      },
      {
        id: 'hparams',
        header: 'Config',
        cell: info => (
          <HparamBadges hparams={info.row.original.hparams} onBadgeClick={onHparamClick} />
        ),
        enableSorting: false,
      },
      {
        id: 'seeds',
        header: 'Seeds',
        cell: info => {
          const r = info.row.original
          return r.isAggregated ? `n=${r.num_seeds}` : `#${r.seed ?? '—'}`
        },
        enableSorting: false,
        size: 64,
      },
      {
        id: 'source',
        header: 'Source',
        cell: info => {
          const r = info.row.original
          const isPaper = r.source_type === 'paper'
          return (
            <span
              title={r.citation || undefined}
              className={`text-xs px-1.5 py-0.5 rounded ${isPaper ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'}`}
            >
              {isPaper ? 'paper' : 'local'}
            </span>
          )
        },
        enableSorting: false,
        size: 72,
      },
    ]

    const metricCols: ColumnDef<DisplayRow>[] = visibleMetrics.map(key => ({
      id: key,
      header: () => (
        <span title={isLowerBetter(key) ? 'lower is better' : 'higher is better'}>
          {key} {isLowerBetter(key) ? '↓' : '↑'}
        </span>
      ),
      accessorFn: (row: DisplayRow) => {
        const m = row.metrics[key]
        return m === undefined ? null : typeof m === 'number' ? m : m.mean
      },
      cell: info => {
        const val = info.row.original.metrics[key]
        if (val === undefined) return <span className="text-gray-300">—</span>
        const mean = typeof val === 'number' ? val : val.mean
        const { best, worst } = bestWorstByMetric[key] ?? { best: NaN, worst: NaN }
        return (
          <MetricCell
            value={val}
            isBest={mean === best}
            isWorst={mean === worst}
            showStd={viewOptions.showStd && info.row.original.isAggregated}
          />
        )
      },
    }))

    return [...staticCols, ...metricCols]
  }, [visibleMetrics, bestWorstByMetric, viewOptions.showStd, onHparamClick])

  const table = useReactTable({
    data: rows,
    columns,
    state: { sorting },
    onSortingChange: (updater) => {
      const next = typeof updater === 'function' ? updater(sorting) : updater
      setSorting(next)
      if (next.length > 0) onSortChange(next[0].id, next[0].desc ? 'desc' : 'asc')
    },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    manualSorting: false,
  })

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm text-left">
        <thead className="bg-gray-50 border-b border-gray-200 sticky top-0">
          {table.getHeaderGroups().map(hg => (
            <tr key={hg.id}>
              {hg.headers.map(header => (
                <th
                  key={header.id}
                  style={{ width: header.getSize() }}
                  className={`px-3 py-2 text-xs font-semibold text-gray-600 whitespace-nowrap ${header.column.getCanSort() ? 'cursor-pointer select-none hover:bg-gray-100' : ''}`}
                  onClick={header.column.getToggleSortingHandler()}
                >
                  {flexRender(header.column.columnDef.header, header.getContext())}
                  {header.column.getIsSorted() === 'asc' ? ' ▲' : header.column.getIsSorted() === 'desc' ? ' ▼' : ''}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody className="divide-y divide-gray-100">
          {table.getRowModel().rows.map(row => (
            <tr key={row.id} className="hover:bg-gray-50">
              {row.getVisibleCells().map(cell => (
                <td key={cell.id} className="px-3 py-2 whitespace-nowrap">
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td colSpan={columns.length} className="px-3 py-8 text-center text-gray-400">
                No results match the current filters.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
```

- [ ] **Step 2: Run all tests**

```bash
cd webapp && npm test -- --run
```

Expected: all existing tests pass.

- [ ] **Step 3: Commit**

```bash
git add webapp/src/components/LeaderboardTable.tsx
git commit -m "feat: add LeaderboardTable with TanStack Table"
```

---

## Task 9: `App.tsx` integration + export CSV

**Files:**
- Modify: `webapp/src/App.tsx`

- [ ] **Step 1: Write the CSV export utility inline in App.tsx**

Helper: given rows and visible metrics, returns a CSV string.

```ts
function rowsToCsv(rows: DisplayRow[], visibleMetrics: string[]): string {
  const metricHeaders = visibleMetrics.flatMap(k => [`${k}_mean`, `${k}_std`])
  const header = ['model', 'task', 'dataset', 'hparams', 'seeds', 'source', ...metricHeaders]
  const lines = rows.map(r => {
    const metricCells = visibleMetrics.flatMap(k => {
      const m = r.metrics[k]
      if (m === undefined) return ['', '']
      if (typeof m === 'number') return [String(m), '']
      return [String(m.mean), String(m.std)]
    })
    return [
      r.model, r.task, r.dataset,
      JSON.stringify(r.hparams),
      String(r.num_seeds),
      r.source_type,
      ...metricCells,
    ].map(v => `"${v.replace(/"/g, '""')}"`).join(',')
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
```

- [ ] **Step 2: Write the full `App.tsx`**

`webapp/src/App.tsx`:
```tsx
import { useState, useMemo, useCallback } from 'react'
import { useLeaderboard } from './hooks/useLeaderboard'
import { useFiltered } from './hooks/useFiltered'
import { FilterSidebar } from './components/FilterSidebar'
import { ToolBar } from './components/ToolBar'
import { LeaderboardTable } from './components/LeaderboardTable'
import type { Filters, ViewOptions, DisplayRow } from './types'

function rowsToCsv(rows: DisplayRow[], visibleMetrics: string[]): string {
  const metricHeaders = visibleMetrics.flatMap(k => [`${k}_mean`, `${k}_std`])
  const header = ['model', 'task', 'dataset', 'hparams', 'seeds', 'source', ...metricHeaders]
  const lines = rows.map(r => {
    const metricCells = visibleMetrics.flatMap(k => {
      const m = r.metrics[k]
      if (m === undefined) return ['', '']
      if (typeof m === 'number') return [String(m), '']
      return [String(m.mean), String(m.std)]
    })
    return [
      r.model, r.task, r.dataset,
      JSON.stringify(r.hparams), String(r.num_seeds), r.source_type,
      ...metricCells,
    ].map(v => `"${String(v).replace(/"/g, '""')}"`).join(',')
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

const isLiveServer = !import.meta.env.PROD || window.location.port !== ''

export default function App() {
  const { data, error, loading, refresh } = useLeaderboard()

  const [filters, setFilters] = useState<Filters>({
    task: null, datasets: [], models: [], hparams: {},
  })

  const allMetrics = useMemo(() => {
    if (!data) return []
    const keys = new Set<string>()
    data.entries.forEach(e => Object.keys(e.metrics).forEach(k => keys.add(k)))
    return Array.from(keys).sort()
  }, [data])

  const [viewOptions, setViewOptions] = useState<ViewOptions>({
    aggregate: true, showStd: false,
    visibleMetrics: allMetrics,
    sortColumn: null, sortDirection: 'asc',
  })

  // keep visibleMetrics in sync when new metrics appear
  const effectiveOptions = useMemo(() => ({
    ...viewOptions,
    visibleMetrics: viewOptions.visibleMetrics.length > 0
      ? viewOptions.visibleMetrics
      : allMetrics,
  }), [viewOptions, allMetrics])

  const rows = useFiltered(data?.entries ?? [], filters, effectiveOptions)

  const handleHparamClick = useCallback((key: string, value: string) => {
    setFilters(f => ({ ...f, hparams: { ...f.hparams, [key]: value } }))
  }, [])

  const handleExportCsv = useCallback(() => {
    downloadCsv(rowsToCsv(rows, effectiveOptions.visibleMetrics), 'leaderboard.csv')
  }, [rows, effectiveOptions.visibleMetrics])

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

  if (!data) return null

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center gap-3">
        <h1 className="text-lg font-bold text-gray-900">torch-timeseries Leaderboard</h1>
        <span className="text-xs text-gray-400">
          Generated {new Date(data.generated_at).toLocaleString()}
        </span>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <FilterSidebar schema={data.schema} filters={filters} onChange={setFilters} />

        {/* Main */}
        <main className="flex-1 flex flex-col overflow-hidden">
          <ToolBar
            viewOptions={effectiveOptions}
            allMetrics={allMetrics}
            resultCount={rows.length}
            onViewChange={setViewOptions}
            onExportCsv={handleExportCsv}
            onRefresh={isLiveServer ? refresh : undefined}
          />
          <div className="flex-1 overflow-auto">
            <LeaderboardTable
              rows={rows}
              viewOptions={effectiveOptions}
              onSortChange={(col, dir) => setViewOptions(v => ({ ...v, sortColumn: col, sortDirection: dir }))}
              onHparamClick={handleHparamClick}
            />
          </div>
        </main>
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Verify dev server shows real data**

First generate the data if not done yet:
```bash
cd /data/yww/notebook/pytorchtimseries
python scripts/build_leaderboard.py
```

Then:
```bash
cd webapp && npm run dev
```

Open http://localhost:5173. Expected: sidebar with task list, table with model rows, filter controls work.

- [ ] **Step 4: Run production build**

```bash
cd webapp && npm run build
```

Expected: `dist/` created, no TypeScript errors, build summary printed.

- [ ] **Step 5: Verify `dist/` self-contains the data**

```bash
ls webapp/dist/ && ls webapp/dist/assets/
```

Expected: `index.html` and `assets/` dir with JS/CSS bundles. `leaderboard_data.json` will be present because it's in `public/`.

- [ ] **Step 6: Run all tests**

```bash
cd webapp && npm test -- --run
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add webapp/src/App.tsx
git commit -m "feat: wire App.tsx — full leaderboard with filter, aggregate, and CSV export"
```

---

## Task 10: Live server + GitHub Actions

**Files:**
- Create: `scripts/serve_leaderboard.py`
- Create: `.github/workflows/leaderboard.yml`

- [ ] **Step 1: Write `scripts/serve_leaderboard.py`**

`scripts/serve_leaderboard.py`:
```python
#!/usr/bin/env python3
"""
Live leaderboard server. Serves webapp/dist/ and provides /api/refresh.

Usage:
    python scripts/serve_leaderboard.py           # port 8000
    python scripts/serve_leaderboard.py --port 9000
"""
import argparse
import pathlib
import subprocess
import sys

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

ROOT = pathlib.Path(__file__).resolve().parent.parent
DIST = ROOT / "webapp" / "dist"
BUILD = ROOT / "scripts" / "build_leaderboard.py"

app = FastAPI(title="Leaderboard Server")


@app.get("/api/refresh")
def refresh():
    result = subprocess.run(
        [sys.executable, str(BUILD)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr.strip())
    return {"status": "ok", "message": result.stdout.strip()}


if not DIST.exists():
    raise RuntimeError(
        f"webapp/dist/ not found. Run: cd webapp && npm run build"
    )

app.mount("/", StaticFiles(directory=str(DIST), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run("scripts.serve_leaderboard:app", host=args.host, port=args.port, reload=False)
```

- [ ] **Step 2: Test the live server**

```bash
cd /data/yww/notebook/pytorchtimseries
python scripts/serve_leaderboard.py &
sleep 2
curl -s http://localhost:8000/api/refresh | python3 -m json.tool
```

Expected: `{"status": "ok", "message": "Wrote N entries → ..."}`. Open http://localhost:8000 in browser.

```bash
kill %1  # stop background server
```

- [ ] **Step 3: Write `.github/workflows/leaderboard.yml`**

`.github/workflows/leaderboard.yml`:
```yaml
name: Deploy Leaderboard

on:
  push:
    branches: [main]
    paths:
      - 'results/**'
      - 'leaderboard/entries/**'
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

- [ ] **Step 4: Commit everything**

```bash
git add scripts/serve_leaderboard.py .github/workflows/leaderboard.yml
git commit -m "feat: add live server and GitHub Actions deploy workflow"
```

---

## Final verification

- [ ] **Run all Python tests**

```bash
python -m pytest tests/leaderboard/ -v
```

Expected: 6 passed

- [ ] **Run all frontend tests**

```bash
cd webapp && npm test -- --run
```

Expected: ≥ 16 passed, 0 failed

- [ ] **Run production build end-to-end**

```bash
python scripts/build_leaderboard.py && cd webapp && npm run build && ls dist/
```

Expected: `index.html`, `assets/`, `leaderboard_data.json` all present in `dist/`
