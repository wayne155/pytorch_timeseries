import torch
from torch import Tensor
from typing import Generic, TypeVar, Union

import pandas as pd
import numpy as np
import torch
StoreType = TypeVar('StoreType')

class Scaler(Generic[StoreType]):
    """
    Generic Scaler class for fitting, transforming, and inverse transforming datasets.

    Methods:
        fit(data: StoreType) -> None:
            Fit the Scaler to the input dataset.

        transform(data: StoreType) -> StoreType:
            Transform the input dataset using the Scaler.

        inverse_transform(data: StoreType) -> StoreType:
            Perform an inverse transform on the input dataset using the Scaler.
    """

    def fit(self, data: StoreType) -> None:
        """
        Fit the Scaler to the input dataset.

        Args:
            data (StoreType): The input dataset to fit the Scaler to.

        Returns:
            None
        """
        raise NotImplementedError()

    def transform(self, data: StoreType) -> StoreType:
        """
        Transform the input dataset using the Scaler.

        Args:
            data (StoreType): The input dataset to transform using the Scaler.

        Returns:
            StoreType: The transformed dataset of the same type as the input data.
        """
        raise NotImplementedError()

    def inverse_transform(self, data: StoreType) -> StoreType:
        """
        Perform an inverse transform on the input dataset using the Scaler.

        Args:
            data (StoreType): The input dataset to perform an inverse transform on.

        Returns:
            StoreType: The inverse transformed dataset of the same type as the input data.
        """
        raise NotImplementedError()

