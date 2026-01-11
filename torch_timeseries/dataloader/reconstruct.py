from typing import Sequence, Tuple, Type

import torch
from ..scaler import Scaler
from torch_timeseries.core import (
    TimeSeriesDataset,
    TimeseriesSubset,
)
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset

from .wrapper import ReconstructSet


class Reconstruct:
    def __init__(
        self,
        dataset: TimeSeriesDataset,
        scaler: Scaler,
        shuffle=True,
        batch_size: int = 32,
        num_worker: int = 3,
    ) -> None:
        """
        Class for splitting the dataset sequentially and then randomly sampling from each subset.

        Attributes:
            dataset (TimeSeriesDataset): Time series dataset to be used.
            scaler (Scaler): Scaler to normalize the data.
            window (int): Window size for the time series data.
            horizon (int): Forecast horizon.
            steps (int): Step size between windows.
            scale_in_train (bool): Whether to scale data during training.
            shuffle_train (bool): Whether to shuffle the training data.
            batch_size (int): Number of samples per batch.
            train_ratio (float): Ratio of the dataset to be used for training.
            val_ratio (float): Ratio of the dataset to be used for validation.
            test_ratio (float): Ratio of the dataset to be used for testing.
            num_worker (int): Number of worker threads for data loading.
            uniform_eval (bool): Whether to use uniform evaluation.
            train_loader (DataLoader): DataLoader for the training data.
            val_loader (DataLoader): DataLoader for the validation data.
            test_loader (DataLoader): DataLoader for the test data.
        """
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.dataset = dataset

        self.scaler = scaler
        self.shuffle = shuffle

        self._load()

    def _load(self):
        self._load_dataset()
        self._load_dataloader()

    def _load_dataset(self):
        """
        Return the splitted training, testing and validation dataloders

        :return: a tuple of train_dataloader, test_dataloader and val_dataloader
        """
        # fixed suquence dataset
        indices = range(0, len(self.dataset))
        train_size = int(0.7*len(self.dataset))
        train_subset = TimeseriesSubset(self.dataset, indices[0:train_size])
        self.scaler.fit(train_subset.data)
        self.recon_set = ReconstructSet(self.dataset, self.scaler)

    def _load_dataloader(self):
        self.dataloader = DataLoader(
            self.recon_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_worker,
        )

