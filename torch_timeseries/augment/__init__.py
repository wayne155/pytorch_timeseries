"""torch_timeseries.augment — composable time series augmentations.

All augmenters are pure-function callables operating on
``(B, T, C)`` float tensors. They can be composed with
:class:`Compose` and applied inside a training loop or a custom
dataset ``__getitem__``.

Example::

    from torch_timeseries.augment import Compose, Jitter, Scale, MagnitudeWarp

    aug = Compose([Jitter(sigma=0.03), Scale(sigma=0.1), MagnitudeWarp(sigma=0.2)])
    x_aug = aug(x)   # x: (B, T, C)
"""

from .transforms import (
    Compose,
    Jitter,
    Scale,
    MagnitudeWarp,
    TimeWarp,
    WindowSlice,
    Permute,
    Flip,
    RandomMask,
)

__all__ = [
    "Compose",
    "Jitter",
    "Scale",
    "MagnitudeWarp",
    "TimeWarp",
    "WindowSlice",
    "Permute",
    "Flip",
    "RandomMask",
]
