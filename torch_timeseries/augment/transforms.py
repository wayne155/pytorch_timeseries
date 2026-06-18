"""Time series augmentation transforms.

All transforms are callables operating on ``(B, T, C)`` or ``(T, C)`` float
tensors.  They are stochastic: different calls may produce different outputs.
Set ``torch.manual_seed`` before applying for reproducibility.

Reference: Um et al., "Data Augmentation of Wearable Sensor Data for Parkinson's
Disease Monitoring using Convolutional Neural Networks", 2017;
Iwana & Uchida, "An empirical survey of data augmentation for time series
classification with neural networks", PLOS ONE 2021.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


class Compose:
    """Apply a sequence of augmentations in order.

    Args:
        transforms: List of augmenters to apply sequentially.

    Example::

        aug = Compose([Jitter(sigma=0.05), Scale(sigma=0.1)])
        x_aug = aug(x)
    """

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class Jitter:
    """Add independent Gaussian noise to every sample.

    Args:
        sigma (float): Standard deviation of the noise. Defaults to 0.05.

    Example::

        aug = Jitter(sigma=0.03)
        x_aug = aug(x)   # x: (B, T, C) or (T, C)
    """

    def __init__(self, sigma: float = 0.05):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.sigma


class Scale:
    """Multiply each sample by a random scalar drawn from N(1, sigma²).

    A separate scalar is drawn per batch item and channel.

    Args:
        sigma (float): Standard deviation of the scaling factor. Defaults to 0.1.
    """

    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:           # (T, C)
            scale = 1 + torch.randn(1, x.shape[-1], device=x.device) * self.sigma
        else:                       # (B, T, C)
            scale = 1 + torch.randn(x.shape[0], 1, x.shape[-1], device=x.device) * self.sigma
        return x * scale


class MagnitudeWarp:
    """Warp amplitudes by multiplying with a smooth random curve.

    A cubic spline with ``n_knots`` randomly-placed knots is sampled per
    batch item and broadcast across channels.

    Args:
        sigma (float): Standard deviation of the knot values. Defaults to 0.2.
        n_knots (int): Number of control points. Defaults to 4.
    """

    def __init__(self, sigma: float = 0.2, n_knots: int = 4):
        self.sigma = sigma
        self.n_knots = n_knots

    def _smooth_curve(self, T: int, n: int, device) -> torch.Tensor:
        # n random knot values around 1.0 then interpolate to length T
        knots = 1.0 + torch.randn(n, device=device) * self.sigma      # (n,)
        knots = knots.unsqueeze(0).unsqueeze(0)                         # (1, 1, n)
        curve = F.interpolate(knots, size=T, mode="linear", align_corners=True)
        return curve.squeeze()                                           # (T,)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            T, C = x.shape
            curve = self._smooth_curve(T, self.n_knots, x.device)      # (T,)
            return x * curve.unsqueeze(-1)
        B, T, C = x.shape
        curves = torch.stack(
            [self._smooth_curve(T, self.n_knots, x.device) for _ in range(B)]
        )  # (B, T)
        return x * curves.unsqueeze(-1)


class TimeWarp:
    """Stretch and compress local regions of the time axis.

    Generates a smooth monotonically-increasing mapping of time indices
    and resamples the series at the warped locations.

    Args:
        sigma (float): Standard deviation of warp displacement. Defaults to 0.2.
        n_knots (int): Number of warp control points. Defaults to 4.
    """

    def __init__(self, sigma: float = 0.2, n_knots: int = 4):
        self.sigma = sigma
        self.n_knots = n_knots

    def _warp_steps(self, T: int, device) -> torch.Tensor:
        tt = torch.ones(self.n_knots, device=device)
        tt = tt + torch.randn(self.n_knots, device=device) * self.sigma
        tt = tt.clamp(min=1e-3)
        steps = torch.cumsum(tt, dim=0)
        steps = steps / steps[-1] * (T - 1)   # rescale to [0, T-1]
        knots = torch.linspace(0, T - 1, self.n_knots, device=device)
        # interpolate to T points
        steps = steps.unsqueeze(0).unsqueeze(0)   # (1, 1, n_knots)
        steps = F.interpolate(steps, size=T, mode="linear", align_corners=True)
        return steps.squeeze().clamp(0, T - 1)    # (T,)

    def _resample(self, x_1d: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # x_1d: (T,), idx: (T,) — gather via bilinear-ish integer lerp
        T = x_1d.shape[0]
        lo = idx.long().clamp(0, T - 2)
        hi = (lo + 1).clamp(0, T - 1)
        frac = (idx - lo.float()).clamp(0, 1)
        return x_1d[lo] * (1 - frac) + x_1d[hi] * frac

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            T, C = x.shape
            idx = self._warp_steps(T, x.device)
            return torch.stack([self._resample(x[:, c], idx) for c in range(C)], dim=-1)
        B, T, C = x.shape
        out = []
        for b in range(B):
            idx = self._warp_steps(T, x.device)
            sample = torch.stack(
                [self._resample(x[b, :, c], idx) for c in range(C)], dim=-1
            )
            out.append(sample)
        return torch.stack(out, dim=0)


class WindowSlice:
    """Extract a contiguous sub-window and interpolate back to the original length.

    Args:
        crop_ratio (float): Fraction of the series to keep, in (0, 1). Defaults to 0.9.
    """

    def __init__(self, crop_ratio: float = 0.9):
        assert 0 < crop_ratio < 1
        self.crop_ratio = crop_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            T, C = x.shape
            L = max(1, int(T * self.crop_ratio))
            start = torch.randint(0, T - L + 1, (1,)).item()
            sliced = x[start: start + L, :]               # (L, C)
            sliced = sliced.T.unsqueeze(0)                 # (1, C, L)
            out = F.interpolate(sliced, size=T, mode="linear", align_corners=True)
            return out.squeeze(0).T                        # (T, C)
        B, T, C = x.shape
        L = max(1, int(T * self.crop_ratio))
        starts = torch.randint(0, T - L + 1, (B,))
        out = []
        for b in range(B):
            s = starts[b].item()
            sliced = x[b, s: s + L, :].T.unsqueeze(0)     # (1, C, L)
            r = F.interpolate(sliced, size=T, mode="linear", align_corners=True)
            out.append(r.squeeze(0).T)                     # (T, C)
        return torch.stack(out, dim=0)


class Permute:
    """Randomly permute equal-length segments along the time axis.

    Args:
        n_segments (int): Number of segments to create and permute. Defaults to 4.
    """

    def __init__(self, n_segments: int = 4):
        self.n_segments = n_segments

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-2]
        seg_len = T // self.n_segments
        if seg_len == 0:
            return x
        idx = torch.randperm(self.n_segments)
        segs = [x[..., i * seg_len: (i + 1) * seg_len, :] for i in idx]
        remainder = x[..., self.n_segments * seg_len:, :]
        parts = segs + ([remainder] if remainder.shape[-2] > 0 else [])
        return torch.cat(parts, dim=-2)


class Flip:
    """Reverse the time axis (temporal reflection).

    Useful for data augmentation when the sequence direction is ambiguous.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.flip(dims=[-2])


class RandomMask:
    """Randomly zero-out time steps with probability ``p``.

    Simulates missing-at-random sensors or irregular observation patterns.

    Args:
        p (float): Probability of masking each time step. Defaults to 0.1.
    """

    def __init__(self, p: float = 0.1):
        assert 0 <= p < 1
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.bernoulli(
            torch.full(x.shape[:-1] + (1,), 1 - self.p, device=x.device)
        )
        return x * mask
