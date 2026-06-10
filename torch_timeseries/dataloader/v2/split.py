"""Train/val/test split configuration, shared by every task datamodule.

A split is either ratio-based or border-based. Datasets with a canonical
benchmark split (the ETT family) are registered in ``DEFAULT_SPLIT_CONFIGS``;
``default_split_config(dataset)`` is what datamodules use when the caller
passes no split config.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SplitConfig:
    train: float = 0.7
    val: Optional[float] = None
    test: Optional[float] = 0.2
    uniform_eval: bool = True
    """If True, val/test subsets are extended by ``window+horizon-1`` so that
    every sample in the original split has a full lookback window."""
    borders: Optional[tuple] = None
    """Explicit ``(train_end, val_end, test_end)`` indices. When set, the
    ratio fields above are ignored. Data past ``test_end`` is unused."""
    use_dataset_borders: bool = True
    """If True and ``borders`` is unset, datasets with a canonical benchmark
    split (see ``DEFAULT_SPLIT_CONFIGS``) use those borders instead of the
    ratios. Set False to force a pure ratio split on such datasets."""


# Canonical benchmark splits for datasets in the repo (TSLib convention).
# ETT: 12 months train / 4 val / 4 test; data past the 20th month is unused.
_ETTH_BORDERS = (12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24)
_ETTM_BORDERS = tuple(b * 4 for b in _ETTH_BORDERS)

DEFAULT_SPLIT_CONFIGS = {
    "ETTh1": SplitConfig(borders=_ETTH_BORDERS),
    "ETTh2": SplitConfig(borders=_ETTH_BORDERS),
    "ETTm1": SplitConfig(borders=_ETTM_BORDERS),
    "ETTm2": SplitConfig(borders=_ETTM_BORDERS),
}


def default_split_config(dataset) -> SplitConfig:
    """The default SplitConfig for *dataset*: its canonical benchmark split
    when registered in ``DEFAULT_SPLIT_CONFIGS``, otherwise 7:1:2 ratios."""
    name = getattr(dataset, "name", type(dataset).__name__)
    return DEFAULT_SPLIT_CONFIGS.get(
        name, SplitConfig(train=0.7, val=0.1, test=0.2)
    )
