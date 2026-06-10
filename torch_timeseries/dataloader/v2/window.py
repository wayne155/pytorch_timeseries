"""Sliding-window configuration, shared by window-based task datamodules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .batch import TimeEncConfig


@dataclass
class WindowConfig:
    window: int = 96
    horizon: int = 1
    steps: int = 96
    stride: int = 1
    fast_val: bool = False
    fast_test: bool = False
    time_enc_cfg: TimeEncConfig = field(default_factory=TimeEncConfig)
    input_columns: Optional[list] = None
    target_columns: Optional[list] = None
