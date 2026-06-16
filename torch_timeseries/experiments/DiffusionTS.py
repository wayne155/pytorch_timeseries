"""Diffusion-TS generation experiment."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.DiffusionTS import DiffusionTS
from ..dataloader.v2.batch import TSBatch
from .generation import GenerationExp


@dataclass
class DiffusionTSGeneration(GenerationExp):
    model_type: str = "DiffusionTS"
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    T: int = 1000
    schedule: str = "cosine"

    def _init_model(self) -> None:
        self.model = DiffusionTS(
            seq_len=self.seq_len,
            n_features=self.num_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            T=self.T,
            schedule=self.schedule,
        ).to(self.device)

    def _process_train_batch(self, batch: TSBatch) -> torch.Tensor:
        return self.model.loss(batch.x.to(self.device))

    def generate(self, n_samples: int, condition=None) -> torch.Tensor:
        self.model.eval()
        return self.model.generate(n_samples, device=self.device)
