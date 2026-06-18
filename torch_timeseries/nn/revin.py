"""Reversible Instance Normalization (RevIN).

Reference: Kim et al., "Reversible Instance Normalization for Accurate
Time-Series Forecasting against Distribution Shift", ICLR 2022.
https://openreview.net/forum?id=cGDAkQo1C0p
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Instance normalization with learnable affine parameters.

    Normalises the *input* window by subtracting its per-channel mean and
    dividing by its std, then re-scales the *output* (or any intermediate
    tensor) by the same statistics in reverse.

    Typical usage in a forecaster::

        revin = RevIN(num_features=C)

        # normalise input
        x_norm = revin(x, mode="norm")       # (B, T, C)

        # run model
        pred_norm = model(x_norm)            # (B, H, C)

        # denormalise forecast
        pred = revin(pred_norm, mode="denorm")  # (B, H, C)

    Args:
        num_features (int): Number of input channels *C*.
        eps (float): Small constant for numerical stability.  Default: 1e-5.
        affine (bool): Whether to learn per-channel scale (γ) and shift (β)
            after normalization.  Default: ``True``.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

        # running statistics — set during forward(mode='norm')
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """Apply normalization or denormalization.

        Args:
            x: Input tensor of shape ``(B, T, C)``.
            mode: ``"norm"`` to normalize and cache statistics;
                  ``"denorm"`` to reverse the cached normalization.

        Returns:
            Transformed tensor of the same shape.
        """
        if mode == "norm":
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        else:
            raise ValueError(f"mode must be 'norm' or 'denorm', got {mode!r}")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) — compute statistics over the time dimension
        self._mean = x.mean(dim=1, keepdim=True)           # (B, 1, C)
        self._std = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + self.eps
        )
        x = (x - self._mean) / self._std
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._std is None:
            raise RuntimeError(
                "RevIN: call forward(x, 'norm') before forward(x, 'denorm')"
            )
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        return x * self._std + self._mean
