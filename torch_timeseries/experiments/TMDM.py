"""TMDM generation experiment (Ye et al.)."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.TMDM import TMDM
from ..dataloader.v2.batch import TSBatch
from .generation import GenerationExp


@dataclass
class TMDMGeneration(GenerationExp):
    model_type: str = "TMDM"
    T: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.5

    def _init_model(self) -> None:
        self.model = TMDM(
            seq_len=self.seq_len,
            n_features=self.num_features,
            T=self.T,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
        ).to(self.device)

    def _process_train_batch(self, batch: TSBatch) -> torch.Tensor:
        return self.model.loss(batch.x.to(self.device))

    def generate(self, n_samples: int, condition=None) -> torch.Tensor:
        self.model.eval()
        return self.model.generate(n_samples, device=self.device)
