"""Point-forecast metrics compatible with torchmetrics.Metric.

All metrics expect flat or broadcastable ``(N,)`` / ``(N, T, C)`` tensors of
predictions and targets.  They integrate with ``MetricCollection`` and
support ``.reset()`` / ``.update()`` / ``.compute()`` semantics.

References
----------
* Makridakis, 1993 — SMAPE definition
* Hyndman & Koehler, 2006 — MASE definition
* Koenker & Bassett, 1978 — Pinball / Quantile loss
"""
from __future__ import annotations

import torch
from torchmetrics import Metric


class SMAPE(Metric):
    """Symmetric Mean Absolute Percentage Error.

    .. math::

        \\text{SMAPE} = \\frac{200}{N} \\sum_{i=1}^{N}
        \\frac{|\\hat{y}_i - y_i|}{|\\hat{y}_i| + |y_i| + \\epsilon}

    The result is a percentage value in ``[0, 200]``.

    Args:
        eps (float): Small constant to avoid division by zero. Default: 1e-8.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.add_state("sum_smape", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = preds.reshape(-1)
        target = target.reshape(-1)
        denom = preds.abs() + target.abs() + self.eps
        self.sum_smape += (200.0 * (preds - target).abs() / denom).sum()
        self.total += target.numel()

    def compute(self) -> torch.Tensor:
        return self.sum_smape / self.total


class MASE(Metric):
    """Mean Absolute Scaled Error.

    Scales the MAE by the in-sample naive (seasonal) forecast error:

    .. math::

        \\text{MASE} = \\frac{\\text{MAE}}{
        \\frac{1}{T - m} \\sum_{t=m+1}^{T} |y_t - y_{t-m}|}

    You must call :meth:`set_naive_mae` once before the first
    :meth:`update` call (or pass ``naive_mae`` to the constructor).

    Args:
        naive_mae (float | None): Pre-computed naive MAE (denominator).  If
            ``None`` you must call :meth:`set_naive_mae` before
            :meth:`compute`.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, naive_mae: float | None = None, **kwargs):
        super().__init__(**kwargs)
        self._naive_mae = naive_mae
        self.add_state("sum_ae", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def set_naive_mae(self, naive_mae: float) -> None:
        self._naive_mae = naive_mae

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.sum_ae += (preds - target).abs().sum()
        self.total += target.numel()

    def compute(self) -> torch.Tensor:
        if self._naive_mae is None:
            raise RuntimeError("Call set_naive_mae() before compute()")
        mae = self.sum_ae / self.total
        return mae / (self._naive_mae + 1e-8)


class QuantileLoss(Metric):
    """Pinball (quantile) loss for probabilistic point estimates.

    .. math::

        L_q(y, \\hat{y}) = q \\cdot \\max(y - \\hat{y}, 0) +
                           (1-q) \\cdot \\max(\\hat{y} - y, 0)

    ``preds`` should be the predicted quantile (not a distribution).

    Args:
        quantile (float): Quantile level in ``(0, 1)``. Default: 0.5
            (median = MAE / 2 by identity).
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, quantile: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        assert 0 < quantile < 1, "quantile must be in (0, 1)"
        self.quantile = quantile
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = preds.reshape(-1)
        target = target.reshape(-1)
        diff = target - preds
        loss = torch.where(diff >= 0, self.quantile * diff,
                           (self.quantile - 1) * diff)
        self.sum_loss += loss.sum()
        self.total += target.numel()

    def compute(self) -> torch.Tensor:
        return self.sum_loss / self.total


def naive_seasonal_mae(y: torch.Tensor, seasonality: int = 1) -> float:
    """Compute the naive seasonal forecast MAE for use as MASE denominator.

    Args:
        y: Training series, shape ``(T,)`` or ``(T, C)``.
        seasonality: Seasonal period *m*.  Use 1 for non-seasonal (random walk).

    Returns:
        Scalar float.
    """
    if y.dim() == 1:
        y = y.unsqueeze(-1)
    T = y.shape[0]
    if T <= seasonality:
        raise ValueError(f"Series length {T} ≤ seasonality {seasonality}")
    return float((y[seasonality:] - y[:-seasonality]).abs().mean())
