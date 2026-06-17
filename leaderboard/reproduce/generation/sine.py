"""Reproduce NsDiff + TMDM generation results on the Sine benchmark."""
from torch_timeseries.experiments.NsDiff import NsDiffGeneration
from torch_timeseries.experiments.TMDM import TMDMGeneration

SEEDS = [1, 2, 3]

if __name__ == "__main__":
    for cls, kwargs in [
        (NsDiffGeneration, dict(T=50, kernel_size=8)),
        (TMDMGeneration,   dict(T=50)),
    ]:
        for seed in SEEDS:
            exp = cls(
                dataset_type="Sine",
                seq_len=24,
                epochs=300,
                batch_size=64,
                eval_n_samples=1000,
                device="cuda:0",
                save_dir="./results/generation",
                **kwargs,
            )
            result = exp.run(seed=seed)
            print(f"{cls.__name__} | Sine | seed={seed} | {result}")
