"""Reproduce CSDI generation results on standard datasets."""
from torch_timeseries.experiments.CSDI import CSDIGeneration

DATASETS = ["ETTh1", "ETTm1", "Weather"]
SEEDS = [1, 2, 3]

if __name__ == "__main__":
    for dataset in DATASETS:
        for seed in SEEDS:
            exp = CSDIGeneration(
                dataset_type=dataset,
                seq_len=24,
                epochs=200,
                d_model=64,
                n_heads=8,
                n_layers=4,
                T=100,
                schedule="linear",
                batch_size=64,
                eval_n_samples=1000,
                device="cuda:0",
                save_dir="./results/generation",
            )
            result = exp.run(seed=seed)
            print(f"CSDI | {dataset} | seed={seed} | {result}")
