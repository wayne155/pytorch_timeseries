#!/usr/bin/env python3
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from leaderboard.reproduce._presets import DEFAULT_SEEDS, LONG_TERM_DATASETS, LONG_TERM_GRID, params_for
from leaderboard.reproduce._runner import run_matrix

MODEL = "NormalizingFlow"
TASK = "Forecast"
DATASETS = LONG_TERM_DATASETS
SEEDS = DEFAULT_SEEDS
GRID = LONG_TERM_GRID
PARAMS = params_for(MODEL, "long_term_forecast")

if __name__ == "__main__":
    run_matrix(MODEL, TASK, DATASETS, SEEDS, PARAMS, GRID)
