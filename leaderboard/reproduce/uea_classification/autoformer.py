#!/usr/bin/env python3
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from leaderboard.reproduce._presets import DEFAULT_SEEDS, NO_GRID, UEA_DATASETS, params_for
from leaderboard.reproduce._runner import run_matrix

MODEL = "Autoformer"
TASK = "UEAClassification"
DATASETS = UEA_DATASETS
SEEDS = DEFAULT_SEEDS
GRID = NO_GRID
PARAMS = params_for(MODEL, "uea_classification")

if __name__ == "__main__":
    run_matrix(MODEL, TASK, DATASETS, SEEDS, PARAMS, GRID)
