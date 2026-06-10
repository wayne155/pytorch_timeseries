"""DataLoader configuration, shared by every task datamodule."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoaderConfig:
    batch_size: int = 32
    num_workers: int = 0
    shuffle_train: bool = True
    pin_memory: bool = False
