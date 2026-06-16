"""CSDI generation experiment."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.CSDI import CSDI
from ..dataloader.v2.batch import TSBatch
from .generation import GenerationExp


@dataclass
class CSDIGeneration(GenerationExp):
    model_type: str = "CSDI"
    d_model: int = 64
    n_heads: int = 8
    n_layers: int = 4
    T: int = 100
    schedule: str = "linear"

    def _init_model(self) -> None:
        self.model = CSDI(
            seq_len=self.seq_len,
            n_features=self.num_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            T=self.T,
            schedule=self.schedule,
        ).to(self.device)

    def _process_train_batch(self, batch: TSBatch) -> torch.Tensor:
        x = batch.x.to(self.device)
        return self.model.loss(x)

    def generate(self, n_samples: int, condition=None) -> torch.Tensor:
        self.model.eval()
        return self.model.generate(n_samples, device=self.device)
