"""Reproduce NS-Diffusion generation results on standard datasets (Ye et al.)."""
from torch_timeseries.experiments.NSDiffusion import NSDiffusionGeneration

DATASETS = ["ETTh1", "ETTm1", "Weather"]
SEEDS = [1, 2, 3]

if __name__ == "__main__":
    for dataset in DATASETS:
        for seed in SEEDS:
            exp = NSDiffusionGeneration(
                dataset_type=dataset,
                seq_len=24,
                epochs=500,
                d_model=128,
                n_heads=4,
                n_layers=4,
                T=500,
                t_emb_dim=64,
                batch_size=64,
                eval_n_samples=1000,
                device="cuda:0",
                save_dir="./results/generation",
            )
            result = exp.run(seed=seed)
            print(f"NSDiffusion | {dataset} | seed={seed} | {result}")
