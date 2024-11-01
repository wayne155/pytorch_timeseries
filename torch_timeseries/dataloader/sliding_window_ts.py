from typing import Sequence, Tuple, Type

import torch
from ..scaler import Scaler
from torch_timeseries.core import (
    TimeSeriesDataset,
    TimeseriesSubset,
)
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset

from .wrapper import MultiStepTimeFeatureSet


class SlidingWindowTS:
    def __init__(
        self,
        dataset: TimeSeriesDataset,
        scaler: Scaler,
        time_enc=3,
        window: int = 168,
        horizon: int = 3,
        steps: int = 2,
        scale_in_train=False,
        shuffle_train=True,
        freq=None,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        num_worker: int = 3,
        uniform_eval=True,
        single_variate=False,
    ) -> None:
        """
        Class for splitting the dataset sequentially and then randomly sampling from each subset.

        Attributes:
            dataset (TimeSeriesDataset): Time series dataset to be used.
            scaler (Scaler): Scaler to normalize the data.
            time_enc (int): Time encoding flag.
            window (int): Window size for the time series data.
            horizon (int): Forecast horizon.
            steps (int): Step size between windows.
            scale_in_train (bool): Whether to scale data during training.
            shuffle_train (bool): Whether to shuffle the training data.
            freq (str or None): Frequency of the time series data.
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
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - self.train_ratio - self.val_ratio
        self.uniform_eval = uniform_eval
        self.single_variate = single_variate

        assert (
            self.train_ratio + self.val_ratio + self.test_ratio == 1.0
        ), "Split ratio must sum up to 1.0"
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.dataset = dataset

        self.scaler = scaler
        self.window = window
        self.freq = freq
        self.time_enc = time_enc
        self.steps = steps
        self.horizon = horizon
        self.shuffle_train = shuffle_train
        self.scale_in_train = scale_in_train

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

        train_size = int(self.train_ratio * len(self.dataset))
        val_size = int(self.val_ratio * len(self.dataset))
        test_size = len(self.dataset) - val_size - train_size
        train_subset = TimeseriesSubset(self.dataset, indices[0:train_size])
        
        if self.uniform_eval:
            val_subset = TimeseriesSubset( # self.window + self.horizon - 1
                self.dataset, indices[train_size - self.window - self.horizon + 1: (val_size + train_size)]
            )
        else:
            val_subset = TimeseriesSubset(
                self.dataset,indices[train_size: (val_size + train_size)]
            )

        if self.uniform_eval:
            test_subset = TimeseriesSubset(self.dataset, indices[-test_size - self.window - self.horizon + 1:])
        else:
            test_subset = TimeseriesSubset(self.dataset, indices[-test_size:])

            
        if self.scale_in_train:
            self.scaler.fit(train_subset.data)
        else:
            self.scaler.fit(self.dataset.data)

        self.train_dataset = MultiStepTimeFeatureSet(
            train_subset,
            scaler=self.scaler,
            time_enc=self.time_enc,
            window=self.window,
            single_variate=self.single_variate,
            horizon=self.horizon,
            steps=self.steps,
            freq=self.freq,
            scaler_fit=False,
        )
        self.val_dataset = MultiStepTimeFeatureSet(
            val_subset,
            scaler=self.scaler,
            time_enc=self.time_enc,
            window=self.window,
            horizon=self.horizon,
            steps=self.steps,
            freq=self.freq,
            single_variate=self.single_variate,
            scaler_fit=False,
        )
        self.test_dataset = MultiStepTimeFeatureSet(
            test_subset,
            scaler=self.scaler,
            time_enc=self.time_enc,
            window=self.window,
            horizon=self.horizon,
            single_variate=self.single_variate,
            steps=self.steps,
            freq=self.freq,
            scaler_fit=False,
        )

    def _load_dataloader(self):
        self.train_size = len(self.train_dataset)
        self.val_size = len(self.val_dataset)
        self.test_size = len(self.test_dataset)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_worker,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
        )

