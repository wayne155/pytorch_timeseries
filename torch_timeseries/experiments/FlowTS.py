"""FlowTS generation experiment.

Time Series Generation via Rectified Flow (Hu et al.).
Training: minimise the rectified flow velocity-matching loss.
Generation: Euler ODE integration from N(0,I) noise to synthetic time series.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.FlowTS import FlowTS
from ..dataloader.v2.batch import TSBatch
from .generation import GenerationExp


@dataclass
class FlowTSGeneration(GenerationExp):
    model_type: str = "FlowTS"
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    n_steps: int = 20

    def _init_model(self) -> None:
        self.model = FlowTS(
            seq_len=self.seq_len,
            n_features=self.num_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            n_steps=self.n_steps,
        ).to(self.device)

    def _process_train_batch(self, batch: TSBatch) -> torch.Tensor:
        x = batch.x.to(self.device)
        return self.model.loss(x)

    def generate(self, n_samples: int, condition=None) -> torch.Tensor:
        self.model.eval()
        return self.model.generate(n_samples, device=self.device, n_steps=self.n_steps)
