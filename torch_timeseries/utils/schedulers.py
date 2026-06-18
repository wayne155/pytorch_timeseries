"""Learning rate schedulers with warmup for time series training loops."""
from __future__ import annotations

import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineScheduler(LRScheduler):
    """Linear warmup followed by cosine annealing down to ``eta_min``.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warmup steps (gradient updates).
        total_steps: Total number of steps including warmup.
        eta_min: Minimum learning rate at the end of cosine decay.
            Default: 0.
        last_epoch: Index of last step (for resuming). Default: -1.

    Example::

        scheduler = WarmupCosineScheduler(
            optimizer, warmup_steps=500, total_steps=10_000
        )
        for batch in loader:
            loss.backward()
            optimizer.step()
            scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps >= 0
        assert total_steps > warmup_steps
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = (step + 1) / max(self.warmup_steps, 1)
        else:
            progress = (step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * scale
            for base_lr in self.base_lrs
        ]


class WarmupLinearScheduler(LRScheduler):
    """Linear warmup followed by linear decay to ``eta_min``.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of steps including warmup.
        eta_min: Minimum learning rate at the end of linear decay.
            Default: 0.
        last_epoch: Index of last step (for resuming). Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps >= 0
        assert total_steps > warmup_steps
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = (step + 1) / max(self.warmup_steps, 1)
        else:
            decay_steps = self.total_steps - self.warmup_steps
            remaining = self.total_steps - step
            scale = max(0.0, remaining / max(decay_steps, 1))
        return [
            self.eta_min + (base_lr - self.eta_min) * scale
            for base_lr in self.base_lrs
        ]
