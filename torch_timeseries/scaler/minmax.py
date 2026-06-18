import numpy as np
import torch
from torch import Tensor

from ..core.scaler import Scaler, StoreType


class MinMaxScaler(Scaler):
    """Scale each feature to the range [``feature_range[0]``, ``feature_range[1]``].

    Fits per-column min and max on the training data; clipping is applied on
    ``transform`` so out-of-range values are clamped to the target range.

    Args:
        feature_range: Target range as a (min, max) tuple. Default: (0, 1).
    """

    def __init__(self, feature_range=(0, 1)) -> None:
        lo, hi = feature_range
        assert lo < hi, "feature_range[0] must be < feature_range[1]"
        self.lo = lo
        self.hi = hi
        self.data_min = None
        self.data_max = None

    def fit(self, data: StoreType) -> None:
        if isinstance(data, np.ndarray):
            self.data_min = np.min(data, axis=0)
            self.data_max = np.max(data, axis=0)
        elif isinstance(data, Tensor):
            self.data_min = data.min(dim=0).values
            self.data_max = data.max(dim=0).values
        else:
            raise ValueError(f"unsupported type: {type(data)}")
        # Avoid division by zero for constant features
        if isinstance(data, np.ndarray):
            rng = self.data_max - self.data_min
            rng[rng == 0] = 1
            self._range = rng
        else:
            rng = self.data_max - self.data_min
            rng[rng == 0] = 1
            self._range = rng

    def transform(self, data: StoreType) -> StoreType:
        if isinstance(data, np.ndarray):
            x = (data - self.data_min) / self._range
            return np.clip(x, 0, 1) * (self.hi - self.lo) + self.lo
        elif isinstance(data, Tensor):
            mn = torch.as_tensor(self.data_min, dtype=data.dtype, device=data.device)
            rng = torch.as_tensor(self._range, dtype=data.dtype, device=data.device)
            x = (data - mn) / rng
            return x.clamp(0, 1) * (self.hi - self.lo) + self.lo
        else:
            raise ValueError(f"unsupported type: {type(data)}")

    def inverse_transform(self, data: StoreType) -> StoreType:
        if isinstance(data, np.ndarray):
            x = (data - self.lo) / (self.hi - self.lo)
            return x * self._range + self.data_min
        elif isinstance(data, Tensor):
            mn = torch.as_tensor(self.data_min, dtype=data.dtype, device=data.device)
            rng = torch.as_tensor(self._range, dtype=data.dtype, device=data.device)
            x = (data - self.lo) / (self.hi - self.lo)
            return x * rng + mn
        else:
            raise ValueError(f"unsupported type: {type(data)}")
