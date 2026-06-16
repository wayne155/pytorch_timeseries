# tests/experiments/test_generation_base.py
import numpy as np
import pandas as pd
import pytest
from torch_timeseries.experiments.generation import GenerationExp
from dataclasses import dataclass
import torch


class _ToyDataset:
    name = "__toy_gen__"
    freq = "h"
    def __init__(self, T=100, C=2):
        self.data = np.random.randn(T, C).astype(np.float32)
        self.num_features = C
        self.length = T
        self.dates = pd.DataFrame(
            {"date": pd.date_range("2020-01-01", periods=T, freq="h")}
        )


@dataclass
class _ToyGenExp(GenerationExp):
    model_type: str = "ToyGen"

    def _init_model(self):
        self.model = torch.nn.Linear(self.num_features, self.num_features).to(self.device)

    def _process_train_batch(self, batch):
        x = batch.x.to(self.device)
        return ((self.model(x) - x) ** 2).mean()

    def generate(self, n_samples, condition=None):
        z = torch.randn(n_samples, self.seq_len, self.num_features)
        return z


def test_run_returns_dict_with_four_metrics(tmp_path):
    exp = _ToyGenExp(
        dataset_type="__toy_gen__",
        seq_len=8,
        epochs=2,
        patience=100,
        batch_size=4,
        eval_n_samples=20,
        device="cpu",
        save_dir=str(tmp_path),
    )
    exp._toy_dataset = _ToyDataset(T=60, C=2)
    result = exp.run(seed=0)
    for key in ("discriminative_score", "predictive_score",
                "context_fid", "correlational_score"):
        assert key in result
        assert isinstance(result[key], float)
        assert np.isfinite(result[key])
