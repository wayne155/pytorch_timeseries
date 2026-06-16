"""TimeDiff generation experiment."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.TimeDiff import TimeDiff
from ..dataloader.v2.batch import TSBatch
from .generation import GenerationExp


@dataclass
class TimeDiffGeneration(GenerationExp):
    model_type: str = "TimeDiff"
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    T: int = 500
    schedule: str = "linear"
    mix_ratio: float = 0.5

    def _init_model(self) -> None:
        self.model = TimeDiff(
            seq_len=self.seq_len,
            n_features=self.num_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            T=self.T,
            schedule=self.schedule,
            mix_ratio=self.mix_ratio,
        ).to(self.device)

    def _process_train_batch(self, batch: TSBatch) -> torch.Tensor:
        return self.model.loss(batch.x.to(self.device))

    def generate(self, n_samples: int, condition=None) -> torch.Tensor:
        self.model.eval()
        return self.model.generate(n_samples, device=self.device)
