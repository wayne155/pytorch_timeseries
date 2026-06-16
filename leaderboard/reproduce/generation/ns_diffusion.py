"""Reproduce NsDiff generation results on standard datasets (Ye et al.)."""
from torch_timeseries.experiments.NsDiff import NsDiffGeneration

DATASETS = ["ETTh1", "ETTm1", "Weather"]
SEEDS = [1, 2, 3]

if __name__ == "__main__":
    for dataset in DATASETS:
        for seed in SEEDS:
            exp = NsDiffGeneration(
                dataset_type=dataset,
                seq_len=96,
                epochs=200,
                T=100,
                kernel_size=24,
                batch_size=64,
                eval_n_samples=1000,
                device="cuda:0",
                save_dir="./results/generation",
            )
            result = exp.run(seed=seed)
            print(f"NsDiff | {dataset} | seed={seed} | {result}")
