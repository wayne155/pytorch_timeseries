#!/usr/bin/env python3
"""Reproduce PatchTST long-term forecast leaderboard results.

    python leaderboard/reproduce/forecast/patchtst.py

Protocol (matches Time-Series-Library):
  split      : dataset default — ETT calendar borders, 12/4/4 months
  optimizer  : Adam, lr 1e-4, halved every epoch (lradj type1)
  training   : 10 epochs, early stop patience 3 on val mse, batch 32
  loss       : MSE on scaled values; metrics reported on scaled values
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from leaderboard.reproduce._runner import run_grid

MODEL = "PatchTST"
TASK = "Forecast"
DATASETS = ["ETTh1"]
PRED_LENS = [96, 192, 336, 720]
SEEDS = [0, 1, 2, 3, 4]

PARAMS = dict(
    # windowing
    windows=96,
    horizon=1,
    time_enc=1,           # timeF encoding, freq from dataset
    # model (PatchTST)
    d_model=512,
    n_heads=8,
    e_layers=1,
    d_ff=2048,
    dropout=0.1,
    patch_len=16,
    stride=8,             # patch stride
    # training (TSLib protocol)
    epochs=10,
    patience=3,
    batch_size=32,
    lr=0.0001,
    lradj="type1",        # lr x0.5 every epoch
)

if __name__ == "__main__":
    run_grid(MODEL, TASK, DATASETS, PRED_LENS, SEEDS, PARAMS)
