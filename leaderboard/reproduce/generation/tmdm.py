"""Reproduce TMDM generation results."""
from torch_timeseries import Experiment

datasets = ["ETTh1", "ETTh2", "Weather"]

for ds in datasets:
    Experiment("TMDM", "Generation", ds, seq_len=96, T=100, epochs=200) \
        .with_local("./results") \
        .run(seeds=[1, 2, 3])
