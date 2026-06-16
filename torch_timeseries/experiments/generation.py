"""Base experiment class for time series generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from torch_timeseries.dataset import *  # noqa: F401,F403  — populate globals for parse_type
from torch_timeseries.scaler import *   # noqa: F401,F403  — populate globals for parse_type

from ..core.experiments.settings import BaseIrrelevant, BaseRelevant
from ..dataloader.v2 import LoaderConfig
from ..dataloader.v2.generation import GenerationDataModule, GenerationWindowConfig
from ..dataloader.v2.split import SplitConfig
from ..dataloader.v2.batch import TSBatch
from ..metrics.generation import (
    discriminative_score,
    predictive_score,
    context_fid,
    correlational_score,
)
from ..utils.parse_type import parse_type
from ..utils.reproduce import reproducible


@dataclass
class GenerationSettings:
    seq_len: int = 96
    train_ratio: Optional[float] = None
    test_ratio: Optional[float] = None


@dataclass
class GenerationExp(BaseRelevant, BaseIrrelevant, GenerationSettings):
    epochs: int = 200
    patience: int = 30
    lradj: str = "cosine"
    eval_n_samples: int = 1000

    # ------------------------------------------------------------------ #
    # dataset                                                             #
    # ------------------------------------------------------------------ #

    def _init_dataset(self) -> None:
        if hasattr(self, "_toy_dataset"):
            self.dataset = self._toy_dataset
            return
        self.dataset = parse_type(self.dataset_type, globals())(root=self.data_path)

    def _init_data_loader(self) -> None:
        self._init_dataset()
        self.scaler = parse_type(self.scaler_type, globals())()

        window_cfg = GenerationWindowConfig(seq_len=self.seq_len)

        if self.train_ratio is None and self.test_ratio is None:
            split_cfg = None
        else:
            split_cfg = SplitConfig(
                train=self.train_ratio if self.train_ratio is not None else 0.7,
                test=self.test_ratio,
            )

        self.datamodule = GenerationDataModule(
            dataset=self.dataset,
            scaler=self.scaler,
            window=window_cfg,
            split=split_cfg,
            loader=LoaderConfig(
                batch_size=self.batch_size,
                num_workers=self.num_worker,
            ),
        )
        self.train_loader = self.datamodule.train_loader
        self.test_loader = self.datamodule.test_loader
        self.num_features = self.datamodule.num_features

    # ------------------------------------------------------------------ #
    # optimizer                                                           #
    # ------------------------------------------------------------------ #

    def _init_optimizer(self) -> None:
        self.model_optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.l2_weight_decay,
        )
        if self.lradj == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.model_optim, T_max=self.epochs
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.model_optim,
                lr_lambda=lambda e: 0.5 ** max(0, e - 1),
            )

    # ------------------------------------------------------------------ #
    # subclass contract                                                   #
    # ------------------------------------------------------------------ #

    def _init_model(self) -> None:
        raise NotImplementedError

    def _process_train_batch(self, batch: TSBatch) -> Tensor:
        """Return a scalar loss for one batch."""
        raise NotImplementedError

    def generate(self, n_samples: int, condition=None) -> Tensor:
        """Return (n_samples, seq_len, num_features) float32 on CPU."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # training / evaluation                                               #
    # ------------------------------------------------------------------ #

    def _train_one_epoch(self) -> float:
        self.model.train()
        losses = []
        for batch in self.train_loader:
            self.model_optim.zero_grad()
            loss = self._process_train_batch(batch)
            loss.backward()
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.model_optim.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    def _collect_real(self) -> Tensor:
        seqs = []
        for batch in self.test_loader:
            seqs.append(batch.x.cpu())
            if sum(s.shape[0] for s in seqs) >= self.eval_n_samples:
                break
        return torch.cat(seqs, dim=0)[: self.eval_n_samples]

    def _evaluate(self) -> dict:
        self.model.eval()
        with torch.no_grad():
            real = self._collect_real()
            fake = self.generate(real.shape[0])
        return {
            "discriminative_score": discriminative_score(real, fake),
            "predictive_score":     predictive_score(real, fake),
            "context_fid":          context_fid(real, fake),
            "correlational_score":  correlational_score(real, fake),
        }

    def run(self, seed: int = 0) -> dict:
        reproducible(seed)
        self._init_data_loader()
        self._init_model()
        self._init_optimizer()

        best_loss = float("inf")
        wait = 0

        with tqdm(total=self.epochs, desc=f"{self.model_type} training") as pbar:
            for _ in range(self.epochs):
                loss = self._train_one_epoch()
                self.scheduler.step()
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss:.4f}")

                if loss < best_loss - 1e-4:
                    best_loss = loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

        return self._evaluate()
