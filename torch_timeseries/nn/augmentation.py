"""Time-series data augmentation modules.

All modules:
- Accept ``(B, L, C)`` tensors (batch, timesteps, channels).
- Are ``nn.Module`` subclasses — apply only during ``model.training``.
- Are differentiable where possible.

References:
    T. T. Um et al., *Data Augmentation of Wearable Sensor Data for
    Parkinson's Disease Monitoring*, 2017.
    B. K. Iwana & S. Uchida, *An Empirical Survey of Data Augmentation for
    Time Series Classification with Neural Networks*, 2021.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class Jitter(nn.Module):
    """Add per-timestep Gaussian noise during training.

    Args:
        sigma (float): Standard deviation of the noise. Defaults to 0.03.

    Shape:
        - Input / Output: ``(B, L, C)``
    """

    def __init__(self, sigma: float = 0.03) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        return x + torch.randn_like(x) * self.sigma


class Scaling(nn.Module):
    """Multiply each sample by a random per-channel scale factor.

    The scale factor is drawn once per ``(B, C)`` pair and broadcast over L.

    Args:
        sigma (float): Standard deviation of the log-normal scale. A value of
            0.1 gives scales roughly in ``[0.9, 1.1]``. Defaults to 0.1.

    Shape:
        - Input / Output: ``(B, L, C)``
    """

    def __init__(self, sigma: float = 0.1) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        scale = 1.0 + torch.randn(x.size(0), 1, x.size(2), device=x.device) * self.sigma
        return x * scale


class Flip(nn.Module):
    """Randomly reverse the time axis of each sample with probability *p*.

    Args:
        p (float): Probability of flipping each sample. Defaults to 0.5.

    Shape:
        - Input / Output: ``(B, L, C)``
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        mask = torch.rand(x.size(0), device=x.device) < self.p  # (B,)
        # Flip the L dimension for selected samples
        out = x.clone()
        out[mask] = x[mask].flip(dims=[1])
        return out


class WindowCutout(nn.Module):
    """Zero out a random contiguous window in each sample.

    The window length is drawn uniformly from ``[min_len, max_len]`` per
    sample.  This is similar to cutout / random erasing for images.

    Args:
        min_len (int): Minimum window length to cut out.
        max_len (int): Maximum window length to cut out.
        p (float): Probability of applying cutout to each sample. Defaults to 0.5.

    Shape:
        - Input / Output: ``(B, L, C)``
    """

    def __init__(self, min_len: int = 5, max_len: int = 20, p: float = 0.5) -> None:
        super().__init__()
        self.min_len = min_len
        self.max_len = max_len
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        B, L, C = x.shape
        out = x.clone()
        for i in range(B):
            if torch.rand(1).item() < self.p:
                cut = torch.randint(self.min_len, self.max_len + 1, (1,)).item()
                cut = min(cut, L)
                start = torch.randint(0, L - cut + 1, (1,)).item()
                out[i, start: start + cut, :] = 0.0
        return out


class MagnitudeWarp(nn.Module):
    """Multiply each time series by a smooth random curve.

    The warp curve is a cubic spline through ``num_knots`` random values
    sampled from ``Normal(1, sigma)``.

    Args:
        sigma (float): Standard deviation of the knot values around 1.
            Defaults to 0.2.
        num_knots (int): Number of knot points. Defaults to 4.

    Shape:
        - Input / Output: ``(B, L, C)``
    """

    def __init__(self, sigma: float = 0.2, num_knots: int = 4) -> None:
        super().__init__()
        self.sigma = sigma
        self.num_knots = num_knots

    def _smooth_curve(self, B: int, L: int, device: torch.device) -> Tensor:
        # Sample knot values (B, num_knots)
        knots = 1.0 + torch.randn(B, self.num_knots, device=device) * self.sigma
        # Linearly interpolate to length L
        curve = torch.nn.functional.interpolate(
            knots.unsqueeze(1),          # (B, 1, num_knots)
            size=L,
            mode="linear",
            align_corners=True,
        ).squeeze(1)                      # (B, L)
        return curve

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        B, L, C = x.shape
        curve = self._smooth_curve(B, L, x.device)  # (B, L)
        return x * curve.unsqueeze(-1)               # (B, L, C)


class TimeWarp(nn.Module):
    """Warp the time axis by a smooth random mapping.

    The warp is constructed from a monotone cumulative sum of positive random
    knot values, then interpolated to the original length.

    Args:
        sigma (float): Controls how much the warp deviates from identity.
            Defaults to 0.2.
        num_knots (int): Number of knot points. Defaults to 4.

    Shape:
        - Input / Output: ``(B, L, C)``
    """

    def __init__(self, sigma: float = 0.2, num_knots: int = 4) -> None:
        super().__init__()
        self.sigma = sigma
        self.num_knots = num_knots

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        B, L, C = x.shape
        # Sample positive increments that sum to L
        steps = torch.abs(
            torch.randn(B, self.num_knots, device=x.device) * self.sigma + 1.0
        ).clamp(min=1e-3)
        steps = steps / steps.sum(dim=1, keepdim=True) * L     # normalize

        # Build warped positions (B, num_knots) as cumulative sum
        warp_knots = torch.cumsum(steps, dim=1) - steps         # starts at 0
        warp_knots = torch.cat(
            [torch.zeros(B, 1, device=x.device), warp_knots], dim=1
        )                                                         # (B, num_knots+1)

        # Interpolate to length L
        warp_indices = torch.nn.functional.interpolate(
            warp_knots.unsqueeze(1),                              # (B, 1, num_knots+1)
            size=L,
            mode="linear",
            align_corners=True,
        ).squeeze(1).clamp(0, L - 1)                             # (B, L)

        # Gather with fractional indices via floor + blend
        idx_lo = warp_indices.long().clamp(0, L - 2)
        idx_hi = (idx_lo + 1).clamp(0, L - 1)
        frac = (warp_indices - idx_lo.float()).unsqueeze(-1)      # (B, L, 1)

        x_lo = torch.gather(x, 1, idx_lo.unsqueeze(-1).expand(B, L, C))
        x_hi = torch.gather(x, 1, idx_hi.unsqueeze(-1).expand(B, L, C))
        return x_lo + frac * (x_hi - x_lo)


class Compose(nn.Module):
    """Apply a sequence of augmentations.

    Args:
        transforms (list[nn.Module]): Augmentation modules to apply in order.

    Shape:
        - Input / Output: ``(B, L, C)``

    Example::

        aug = Compose([Jitter(0.05), Scaling(0.1), WindowCutout(5, 20)])
        x_aug = aug(x)
    """

    def __init__(self, transforms: list[nn.Module]) -> None:
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x: Tensor) -> Tensor:
        for t in self.transforms:
            x = t(x)
        return x
