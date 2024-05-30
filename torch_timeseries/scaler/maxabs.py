import torch
from torch import Tensor
from typing import Generic, TypeVar, Union
from ..core.scaler import Scaler, StoreType
import pandas as pd
import numpy as np
import torch

class MaxAbsScaler(Scaler[StoreType]):
    """
    shape of data :  (N , n)
    - N : sample num
    - n : node num
    Transforms each channel to the range [0, 1].
    """

    def __init__(self) -> None:
        self.scale = None

    def fit(self, data: StoreType):
        if isinstance(data, np.ndarray):
            self.scale = np.max(np.abs(data), axis=0)
        elif isinstance(data, Tensor):
            self.scale = data.abs().max(axis=0).values
        else:
            raise ValueError(f"not supported type : {type(data)}")

    def transform(self, data) -> StoreType:
        # (b , n)  or (n)
        return data / self.scale

    def inverse_transform(self, data: StoreType) -> StoreType:
        if isinstance(data, np.ndarray):
            return data * self.scale
        elif isinstance(data, Tensor):
            return data * torch.tensor(self.scale, device=data.device)
        else:
            raise ValueError(f"not supported type : {type(data)}")
