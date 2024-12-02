from typing import Sequence, Tuple, Type

import torch
from ..scaler import Scaler
from torch_timeseries.core import (
    TimeSeriesDataset,
    TimeseriesSubset,
)
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset

from .wrapper import MultiStepTimeFeatureSet


class ETTHLoader:
    def __init__(
        self,
        dataset: TimeSeriesDataset,
        scaler: Scaler,
        time_enc=0,
        window: int = 168,
        horizon: int = 3,
        steps: int = 2,
        shuffle_train=True,
        freq=None,
        batch_size: int = 32,
        num_worker: int = 3,
    ) -> None:


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
        
        border1s = [0, 12*30*24 - self.window + self.horizon - 1, 12*30*24+4*30*24 - self.window + self.horizon - 1]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        
        # indices = border2s[-1]
        # train_size = len(train_indices)
        # val_size =   len(val_indices)
        # test_size = len(test_indices)
        
        train_indices = list(range(border1s[0], border2s[0]))
        val_indices = list(range(border1s[1], border2s[1]))
        test_indices = list(range(border1s[2], border2s[2]))
        
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)
        
        train_subset = TimeseriesSubset(self.dataset, train_indices)
        val_subset = TimeseriesSubset(self.dataset, val_indices)
        test_subset = TimeseriesSubset(self.dataset, test_indices)
            
        self.scaler.fit(train_subset.data)

        self.train_dataset = MultiStepTimeFeatureSet(
            train_subset,
            scaler=self.scaler,
            time_enc=self.time_enc,
            window=self.window,
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
            scaler_fit=False,
        )
        self.test_dataset = MultiStepTimeFeatureSet(
            test_subset,
            scaler=self.scaler,
            time_enc=self.time_enc,
            window=self.window,
            horizon=self.horizon,
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


class ETTMLoader:
    def __init__(
        self,
        dataset: TimeSeriesDataset,
        scaler: Scaler,
        time_enc=0,
        window: int = 168,
        horizon: int = 3,
        steps: int = 2,
        shuffle_train=True,
        freq=None,
        batch_size: int = 32,
        num_worker: int = 3,
    ) -> None:


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
        border1s = [0, 12*30*24*4 - self.window + self.horizon - 1, 12*30*24*4+4*30*24*4 - self.window + self.horizon - 1]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        
        # indices = border2s[-1]
        # train_size = len(train_indices)
        # val_size =   len(val_indices)
        # test_size = len(test_indices)
        
        train_indices = list(range(border1s[0], border2s[0]))
        val_indices = list(range(border1s[1], border2s[1]))
        test_indices = list(range(border1s[2], border2s[2]))
        
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)
        
        train_subset = TimeseriesSubset(self.dataset, train_indices)
        val_subset = TimeseriesSubset(self.dataset, val_indices)
        test_subset = TimeseriesSubset(self.dataset, test_indices)
            
        self.scaler.fit(train_subset.data)

        self.train_dataset = MultiStepTimeFeatureSet(
            train_subset,
            scaler=self.scaler,
            time_enc=self.time_enc,
            window=self.window,
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
            scaler_fit=False,
        )
        self.test_dataset = MultiStepTimeFeatureSet(
            test_subset,
            scaler=self.scaler,
            time_enc=self.time_enc,
            window=self.window,
            horizon=self.horizon,
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

