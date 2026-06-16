"""Reproduce TimeGAN generation results on standard datasets."""
from torch_timeseries.experiments.TimeGAN import TimeGANGeneration

DATASETS = ["ETTh1", "ETTm1", "Weather"]
SEEDS = [1, 2, 3]

if __name__ == "__main__":
    for dataset in DATASETS:
        for seed in SEEDS:
            exp = TimeGANGeneration(
                dataset_type=dataset,
                seq_len=24,
                epochs=200,
                hidden_dim=24,
                n_layers=3,
                epochs_ae=200,
                epochs_sup=200,
                epochs_joint=50,
                batch_size=128,
                eval_n_samples=1000,
                device="cuda:0",
                save_dir="./results/generation",
            )
            result = exp.run(seed=seed)
            print(f"TimeGAN | {dataset} | seed={seed} | {result}")
