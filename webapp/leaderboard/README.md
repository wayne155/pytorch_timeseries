# torch-timeseries Leaderboard

A static webapp for comparing model results across tasks, datasets, and prediction-length variants. Displays per-task views with aggregated mean±std metrics.

## Quick start

```bash
# 1. Generate the data file from your results
python leaderboard/build_leaderboard.py

# 2. Install frontend deps (first time only)
cd webapp && npm install

# 3. Start dev server
npm run dev
# → http://localhost:5173
```

## Deploy: live local server

Serves the built frontend and lets you refresh data without restarting.

```bash
# Build once
python leaderboard/build_leaderboard.py
cd webapp && npm run build

# Run server (port 8000 by default)
python leaderboard/serve_leaderboard.py
# → http://localhost:8000

# Custom port
python leaderboard/serve_leaderboard.py --port 9000
```

Click **↺ Refresh** in the UI to re-run the build script and reload without restarting the server.

## Deploy: GitHub Pages (static)

Push to `main` — GitHub Actions builds and deploys automatically whenever `results/`, `leaderboard/`, `leaderboard_results/`, `webapp/`, `leaderboard/build_leaderboard.py`, or `leaderboard.yaml` changes.

To deploy manually:

```bash
python leaderboard/build_leaderboard.py
cd webapp && npm run build
# push webapp/dist/ to your gh-pages branch
```

## Adding results

**Benchmark results** — drop files under `leaderboard_results/{Model}/{Task}/{Dataset}/seed{N}/metrics.json`:

```json
{
  "model": "PatchTST",
  "task": "Forecast",
  "dataset": "ETTh1",
  "seed": 1,
  "hparams": {"windows": 96, "pred_len": 96},
  "metrics": {"mse": 0.38, "mae": 0.28}
}
```

**Paper baselines** — add a YAML entry in `leaderboard/entries/`:

```yaml
- model: DLinear
  task: Forecast
  dataset: ETTh1
  hparams: {windows: 96, pred_len: 96}
  metrics: {mse: 0.40, mae: 0.30}
  source_type: paper
  citation: "DLinear (Zeng 2023)"
  url: https://arxiv.org/abs/2205.13504
```

Re-run `python leaderboard/build_leaderboard.py` (or click Refresh in the UI) to pick up new entries.
