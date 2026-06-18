"""Reproducibility utilities."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    After calling this function every random number generator (Python's
    ``random``, NumPy, and PyTorch CPU/CUDA) is seeded with ``seed``.

    Args:
        seed (int): The random seed to use.
        deterministic (bool): If ``True``, sets
            ``torch.backends.cudnn.deterministic = True`` and
            ``torch.backends.cudnn.benchmark = False``.  This eliminates
            non-determinism from cuDNN but may reduce GPU throughput.
            Defaults to ``False``.

    Example::

        from torch_timeseries.utils import set_seed

        set_seed(42)
        # All subsequent random operations are reproducible.

        set_seed(42, deterministic=True)  # fully reproducible on GPU too
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
