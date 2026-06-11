"""Metrics for probabilistic (sample-based) forecasting.

Every metric consumes ensemble predictions of shape ``(B, O, N, S)`` —
batch, prediction length, variables, samples — and truths of shape
``(B, O, N)``.
"""
from __future__ import annotations

import torch
from torchmetrics import Metric


def _crps_ensemble(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Sample-based CRPS estimator, vectorized.

    CRPS(F, y) ≈ E|X − y| − ½·E|X − X′| for ensemble members X.

    Args:
        pred: (M, S) ensemble members per sample.
        true: (M,) observed values.
    Returns:
        (M,) per-sample CRPS.
    """
    m = pred.shape[-1]
    abs_err = (pred - true.unsqueeze(-1)).abs().mean(dim=-1)
    # E|X - X'| via the sorted-samples identity:
    # sum_{i,j}|x_i - x_j| = 2 * sum_k (2k - S + 1) * x_(k)   (k 0-based, asc)
    sorted_pred, _ = torch.sort(pred, dim=-1)
    coeff = 2 * torch.arange(m, device=pred.device, dtype=pred.dtype) - m + 1
    spread = (sorted_pred * coeff).sum(dim=-1) / (m * m)
    return abs_err - spread


class CRPS(Metric):
    """Continuous Ranked Probability Score, averaged per scalar value."""

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, true: torch.Tensor):
        pred = pred.reshape(-1, pred.shape[-1]).float()
        true = true.reshape(-1).float()
        self.total += _crps_ensemble(pred, true).sum()
        self.count += true.numel()

    def compute(self):
        return self.total / self.count


class CRPSSum(Metric):
    """CRPS of the series summed over variables (CRPS-sum)."""

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, true: torch.Tensor):
        pred = pred.sum(dim=2)   # (B, O, S)
        true = true.sum(dim=2)   # (B, O)
        pred = pred.reshape(-1, pred.shape[-1]).float()
        true = true.reshape(-1).float()
        self.total += _crps_ensemble(pred, true).sum()
        self.count += true.numel()

    def compute(self):
        return self.total / self.count


class PICP(Metric):
    """Prediction Interval Coverage Probability for a central interval."""

    def __init__(self, low_percentile: int = 5, high_percentile: int = 95,
                 dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.add_state("covered", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, true: torch.Tensor):
        pred = pred.reshape(-1, pred.shape[-1]).float()
        true = true.reshape(-1).float()
        q = torch.tensor(
            [self.low_percentile / 100.0, self.high_percentile / 100.0],
            device=pred.device, dtype=pred.dtype,
        )
        ci = torch.quantile(pred, q, dim=1)            # (2, M)
        in_range = (true >= ci[0]) & (true <= ci[1])
        self.covered += in_range.float().sum()
        self.count += true.numel()

    def compute(self):
        return self.covered / self.count


class QICE(Metric):
    """Quantile Interval Coverage Error (Han et al., CARD 2022).

    Mean absolute deviation between the empirical share of truths falling in
    each predicted quantile bin and the ideal ``1/n_bins``.
    """

    def __init__(self, n_bins: int = 10, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.n_bins = n_bins
        self.add_state("bin_counts", default=torch.zeros(n_bins), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, true: torch.Tensor):
        pred = pred.reshape(-1, pred.shape[-1]).float()
        true = true.reshape(-1).float()
        q = torch.linspace(0, 1, self.n_bins + 1, device=pred.device, dtype=pred.dtype)
        quantiles = torch.quantile(pred, q, dim=1)               # (n_bins+1, M)
        membership = (true.unsqueeze(0) > quantiles).sum(dim=0)  # (M,) in 0..n_bins+1
        # clamp outliers into the first / last bin
        membership = membership.clamp(min=1, max=self.n_bins) - 1
        self.bin_counts += torch.bincount(
            membership, minlength=self.n_bins
        ).to(self.bin_counts)
        self.count += true.numel()

    def compute(self):
        ratio = self.bin_counts / self.count
        ideal = torch.ones_like(ratio) / self.n_bins
        return (ratio - ideal).abs().mean()


class _PointFromSamples(Metric):
    """Base for point metrics computed on the ensemble mean."""

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def _error(self, point_pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def update(self, pred: torch.Tensor, true: torch.Tensor):
        point = pred.float().mean(dim=-1)               # (B, O, N)
        self.total += self._error(point, true.float()).sum()
        self.count += true.numel()


class ProbMSE(_PointFromSamples):
    def _error(self, point_pred, true):
        return (point_pred - true) ** 2

    def compute(self):
        return self.total / self.count


class ProbMAE(_PointFromSamples):
    def _error(self, point_pred, true):
        return (point_pred - true).abs()

    def compute(self):
        return self.total / self.count


class ProbRMSE(_PointFromSamples):
    def _error(self, point_pred, true):
        return (point_pred - true) ** 2

    def compute(self):
        return torch.sqrt(self.total / self.count)
