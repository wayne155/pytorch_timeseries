import numpy as np
import torch
from torch import Tensor

from ..core.scaler import Scaler, StoreType


class RobustScaler(Scaler):
    """Scale using per-feature median and interquartile range.

    Robust to outliers because it uses the IQR instead of the standard
    deviation.  The transform subtracts the median and divides by the IQR.

    Args:
        quantile_range: ``(q_low, q_high)`` as percentiles in [0, 100].
            Default: ``(25, 75)`` (standard IQR).
    """

    def __init__(self, quantile_range=(25, 75)) -> None:
        q_lo, q_hi = quantile_range
        assert 0 <= q_lo < q_hi <= 100
        self.q_lo = q_lo / 100.0
        self.q_hi = q_hi / 100.0
        self.center = None
        self.scale_ = None

    def fit(self, data: StoreType) -> None:
        if isinstance(data, np.ndarray):
            self.center = np.median(data, axis=0)
            lo = np.quantile(data, self.q_lo, axis=0)
            hi = np.quantile(data, self.q_hi, axis=0)
            iqr = hi - lo
            iqr[iqr == 0] = 1
            self.scale_ = iqr
        elif isinstance(data, Tensor):
            self.center = data.median(dim=0).values
            q_lo_t = torch.quantile(data.float(), self.q_lo, dim=0)
            q_hi_t = torch.quantile(data.float(), self.q_hi, dim=0)
            iqr = q_hi_t - q_lo_t
            iqr[iqr == 0] = 1
            self.scale_ = iqr
        else:
            raise ValueError(f"unsupported type: {type(data)}")

    def transform(self, data: StoreType) -> StoreType:
        if isinstance(data, np.ndarray):
            return (data - self.center) / self.scale_
        elif isinstance(data, Tensor):
            c = torch.as_tensor(self.center, dtype=data.dtype, device=data.device)
            s = torch.as_tensor(self.scale_, dtype=data.dtype, device=data.device)
            return (data - c) / s
        else:
            raise ValueError(f"unsupported type: {type(data)}")

    def inverse_transform(self, data: StoreType) -> StoreType:
        if isinstance(data, np.ndarray):
            return data * self.scale_ + self.center
        elif isinstance(data, Tensor):
            c = torch.as_tensor(self.center, dtype=data.dtype, device=data.device)
            s = torch.as_tensor(self.scale_, dtype=data.dtype, device=data.device)
            return data * s + c
        else:
            raise ValueError(f"unsupported type: {type(data)}")
