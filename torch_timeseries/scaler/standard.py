import torch
from torch import Tensor
from typing import Generic, TypeVar, Union
from ..core.scaler import Scaler, StoreType
import pandas as pd
import numpy as np
import torch


class StandardScaler(Scaler):
    """
    shape of data :  (N , n)
    - N : sample num
    - n : node num
    Transforms each channel to the range [0, 1].
    """

    def __init__(self, device="cpu") -> None:
        self.mean = None
        self.std = None
        self.device = device

    def fit(self, data: StoreType):
        if isinstance(data, np.ndarray):
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
            # do not normalize all zero values
            self.std[self.std == 0] = 1
        elif isinstance(data, Tensor):
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)
            self.std[self.std == 0] = 1
        else:
            raise ValueError(f"not supported type : {type(data)}")

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data: StoreType) -> StoreType:
        if isinstance(data, np.ndarray):
            return data * self.std + self.mean
        elif isinstance(data, Tensor):
            return data * torch.tensor(self.std, device=data.device) + torch.tensor(
                self.mean, device=data.device
            )
        else:
            raise ValueError(f"not supported type : {type(data)}")
