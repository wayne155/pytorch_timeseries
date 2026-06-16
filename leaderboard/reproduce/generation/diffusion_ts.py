"""Reproduce Diffusion-TS generation results on standard datasets."""
from torch_timeseries.experiments.DiffusionTS import DiffusionTSGeneration

DATASETS = ["ETTh1", "ETTm1", "Weather"]
SEEDS = [1, 2, 3]

if __name__ == "__main__":
    for dataset in DATASETS:
        for seed in SEEDS:
            exp = DiffusionTSGeneration(
                dataset_type=dataset,
                seq_len=24,
                epochs=500,
                d_model=128,
                n_heads=4,
                n_layers=4,
                T=1000,
                schedule="cosine",
                batch_size=64,
                eval_n_samples=1000,
                device="cuda:0",
                save_dir="./results/generation",
            )
            result = exp.run(seed=seed)
            print(f"DiffusionTS | {dataset} | seed={seed} | {result}")
