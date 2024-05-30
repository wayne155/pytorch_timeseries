from typing import Sequence, Tuple, Type

import numpy as np
import torch

from torch_timeseries.utils.timefeatures import time_features
from ..scaler import Scaler
from torch_timeseries.core import (
    TimeSeriesDataset,
    TimeseriesSubset,
)
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset

from .wrapper import MultiStepTimeFeatureSet




class MaskTimeFeatureSet(Dataset):
    def __init__(self, dataset: TimeseriesSubset, scaler: Scaler, mask_rate=0.4, time_enc=0, window: int = 168, freq=None, scaler_fit=True):
        self.dataset = dataset
        self.window = window
        self.time_enc = time_enc
        self.scaler = scaler
        self.mask_rate = mask_rate
        self.num_features = self.dataset.num_features
        self.length = self.dataset.length
        if freq is None:
            self.freq = self.dataset.freq
        else:
            self.freq = freq
        if scaler_fit:
            self.scaler.fit(self.dataset.data)
        self.scaled_data = self.scaler.transform(self.dataset.data)
        self.date_enc_data = time_features(
            self.dataset.dates, self.time_enc, self.freq)
        assert len(self.dataset) - self.window  > 0, "Dataset is not long enough!!!"


    def transform(self, values):
        return self.scaler.transform(values)

    def inverse_transform(self, values):
        return self.scaler.inverse_transform(values)

    def __getitem__(self, index):
        if isinstance(index, int):
            scaled_x = self.scaled_data[index:index+self.window]
            
            x = self.dataset.data[index:index+self.window]

            x_date_enc = self.date_enc_data[index:index+self.window]
            
            mask = np.random.rand(self.window, self.dataset.num_features)
            mask[mask <= self.mask_rate] = 0
            mask[mask > self.mask_rate] = 1  
            masked_scaled_x = scaled_x*mask

            return masked_scaled_x, scaled_x, x , mask, x_date_enc
        else:
            raise TypeError('Not surpported index type!!!')

    def __len__(self):
        return len(self.dataset) - self.window 





class MaskTS:
    """
    Data loader for imputation time series datasets.

    Attributes:
        dataset (TimeSeriesDataset): Time series dataset to be used.
        scaler (Scaler): Scaler to normalize the data.
        time_enc (int): Time encoding flag.
        window (int): Window size for the time series data.
        mask_rate (float): Rate at which data is masked.
        scale_in_train (bool): Whether to scale data during training.
        shuffle_train (bool): Whether to shuffle the training data.
        freq (str or None): Frequency of the time series data.
        batch_size (int): Number of samples per batch.
        train_ratio (float): Ratio of the dataset to be used for training.
        val_ratio (float): Ratio of the dataset to be used for validation.
        num_worker (int): Number of worker threads for data loading.
        uniform_eval (bool): Whether to use uniform evaluation.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        test_loader (DataLoader): DataLoader for the test data.
    """
    def __init__(
        self,
        dataset: TimeSeriesDataset,
        scaler: Scaler,
        time_enc=0,
        window: int = 168,
        mask_rate=0.4,
        scale_in_train=True,
        shuffle_train=True,
        freq=None,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        num_worker: int = 3,
        uniform_eval=True,
    ) -> None:

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - self.train_ratio - self.val_ratio
        self.uniform_eval = uniform_eval
        self.mask_rate = mask_rate

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
            val_subset = TimeseriesSubset( 
                self.dataset, indices[train_size - self.window: (val_size + train_size)]
            )
        else:
            val_subset = TimeseriesSubset(
                self.dataset,indices[train_size: (val_size + train_size)]
            )

        if self.uniform_eval:
            test_subset = TimeseriesSubset(self.dataset, indices[-test_size - self.window:])
        else:
            test_subset = TimeseriesSubset(self.dataset, indices[-test_size:])

        if self.scale_in_train:
            self.scaler.fit(train_subset.data)
        else:
            self.scaler.fit(self.dataset.data)

        self.train_dataset = MaskTimeFeatureSet(
            train_subset,
            scaler=self.scaler,
            time_enc=self.time_enc,
            window=self.window,
            freq=self.freq,
            scaler_fit=False,
            mask_rate=self.mask_rate
        )
        self.val_dataset = MaskTimeFeatureSet(
            val_subset,
            scaler=self.scaler,
            time_enc=self.time_enc,
            window=self.window,
            freq=self.freq,
            scaler_fit=False,
            mask_rate=self.mask_rate
            
        )
        self.test_dataset = MaskTimeFeatureSet(
            test_subset,
            scaler=self.scaler,
            time_enc=self.time_enc,
            window=self.window,
            freq=self.freq,
            scaler_fit=False,
            mask_rate=self.mask_rate
            
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

