# torch-timeseries Leaderboard

A static webapp for comparing model results across tasks, datasets, and hparam configs. Supports per-seed rows and aggregated mean±std view.

## Quick start

```bash
# 1. Generate the data file from your results
python scripts/build_leaderboard.py

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
python scripts/build_leaderboard.py
cd webapp && npm run build

# Run server (port 8000 by default)
python scripts/serve_leaderboard.py
# → http://localhost:8000

# Custom port
python scripts/serve_leaderboard.py --port 9000
```

Click **↺ Refresh** in the UI to re-run the build script and reload without restarting the server.

## Deploy: GitHub Pages (static)

Push to `main` — GitHub Actions builds and deploys automatically whenever `results/`, `leaderboard/entries/`, `webapp/`, or `scripts/build_leaderboard.py` changes.

To deploy manually:

```bash
python scripts/build_leaderboard.py
cd webapp && npm run build
# push webapp/dist/ to your gh-pages branch
```

## Adding results

**From a local run** — drop a JSON file in `results/`:

```json
{
  "model": "PatchTST",
  "task": "Forecast",
  "dataset": "ETTh1",
  "seed": 1,
  "hparams": {"windows": 96, "pred_len": 96, "horizon": 1},
  "metrics": {"mse": 0.38, "mae": 0.28}
}
```

**From a paper** — add a YAML entry in `leaderboard/entries/`:

```yaml
- model: DLinear
  task: Forecast
  dataset: ETTh1
  hparams: {windows: 96, pred_len: 96, horizon: 1}
  metrics: {mse: 0.40, mae: 0.30}
  source_type: paper
  citation: "DLinear (Zeng 2023)"
  url: https://arxiv.org/abs/2205.13504
```

Re-run `python scripts/build_leaderboard.py` (or click Refresh) to pick up new entries.

## Hparam keys shown per task

Only task-relevant hparams appear in the filter sidebar — infrastructure keys like `device`, `batch_size`, `lr` are stripped automatically.

| Task | Hparams shown |
|------|--------------|
| Forecast | windows, pred_len, horizon |
| UEAClassification | windows |
| AnomalyDetection | windows |
| Imputation | windows, mask_rate |
| IrregularClassification | windows |
| Generation | seq_len |
| ProbForecast | seq_len, pred_len |
