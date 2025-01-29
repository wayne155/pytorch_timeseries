from typing import Sequence, Tuple, Type

import torch
from ..scaler import Scaler
from torch_timeseries.core import (
    TimeSeriesDataset,
    TimeseriesSubset,
)
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset

from .wrapper import MultiStepTimeFeatureSet
from .sliding_window_ts import SlidingWindowTS


class ETTHLoader(SlidingWindowTS):
    def __init__(
        self,
        dataset: TimeSeriesDataset,
        scaler: Scaler,
        time_enc=3,
        window: int = 168,
        horizon: int = 3,
        steps: int = 2,
        shuffle_train=True,
        freq=None,
        batch_size: int = 32,
        num_worker: int = 3,
        single_variate=False,
        fast_test=True,
        fast_val=True,
    ) -> None:

        self.fast_test = fast_test
        self.fast_val = fast_val
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.dataset = dataset
        self.single_variate = single_variate

        self.scaler = scaler
        self.window = window
        self.freq = freq
        self.time_enc = time_enc
        self.steps = steps
        self.horizon = horizon
        self.shuffle_train = shuffle_train

        self._load()

    def _load_subset(self):
        """
        Return the splitted training, testing and validation dataloders

        :return: a tuple of train_dataloader, test_dataloader and val_dataloader
        """
        # fixed suquence dataset
        border1s = [0, 12*30*24 - self.window - self.horizon + 1, 12*30*24+4*30*24 - self.window - self.horizon + 1]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        
        train_indices = list(range(border1s[0], border2s[0]))
        val_indices = list(range(border1s[1], border2s[1]))
        test_indices = list(range(border1s[2], border2s[2]))
        
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)
        
        self.train_subset = TimeseriesSubset(self.dataset, train_indices)
        self.val_subset = TimeseriesSubset(self.dataset, val_indices)
        self.test_subset = TimeseriesSubset(self.dataset, test_indices)
            
        self.scaler.fit(self.train_subset.data)



class ETTMLoader(SlidingWindowTS):
    def __init__(
        self,
        dataset: TimeSeriesDataset,
        scaler: Scaler,
        time_enc=3,
        window: int = 168,
        horizon: int = 3,
        steps: int = 2,
        shuffle_train=True,
        freq=None,
        batch_size: int = 32,
        num_worker: int = 3,
        single_variate=False,
        fast_test=False,
        fast_val=False,
    ) -> None:

        self.fast_test = fast_test
        self.fast_val = fast_val
        self.single_variate = single_variate
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

    def _load_subset(self):
        """
        Return the splitted training, testing and validation dataloders

        :return: a tuple of train_dataloader, test_dataloader and val_dataloader
        """
        # fixed suquence dataset
        border1s = [0, 12*30*24*4 - self.window - self.horizon + 1, 12*30*24*4+4*30*24*4 - self.window - self.horizon + 1]
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
        
        self.train_subset = TimeseriesSubset(self.dataset, train_indices)
        self.val_subset = TimeseriesSubset(self.dataset, val_indices)
        self.test_subset = TimeseriesSubset(self.dataset, test_indices)
            
        self.scaler.fit(self.train_subset.data)

