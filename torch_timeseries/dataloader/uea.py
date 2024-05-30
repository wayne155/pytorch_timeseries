from typing import Sequence, Tuple, Type

import numpy as np
import torch

from torch_timeseries.dataset import UEA
from torch_timeseries.utils.timefeatures import time_features
from ..scaler import Scaler
from torch_timeseries.core import (
    TimeSeriesDataset,
    TimeseriesSubset,
)
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset

from .wrapper import MultiStepTimeFeatureSet





class UEADataset(Dataset):
    def __init__(self, dataset: UEA, scaler: Scaler, flag: str ='train', window: int = 168, scaler_fit=True):
        self.dataset = dataset
        self.window = window
        self.scaler = scaler
        self.flag = flag
        if self.flag == 'test':
            self.scaled_feature_df = self.scaler.transform(self.dataset.test_df)
            self.feature_df = self.dataset.test_df
            self.labels = self.dataset.test_labels
        elif self.flag == 'train':
            if scaler_fit:
                self.scaler.fit(self.dataset.train_features_data)

            self.scaled_feature_df = self.scaler.transform(self.dataset.train_df)
            self.feature_df = self.dataset.train_df
            self.labels = self.dataset.train_labels

        self.indexes = self.feature_df.index.unique()
        
    def __getitem__(self, ind):
        scaled_x = torch.tensor(self.scaled_feature_df.loc[ind].values)
        x = torch.tensor(self.feature_df.loc[ind].values)
        y = torch.tensor(self.labels.loc[ind].values)
        return scaled_x, x, y
      
    def __len__(self):
        return len(self.indexes)




class UEAClassification:
    """
    Class for handling the classification of UEA datasets.

    Attributes:
        batch_size (int): Number of samples per batch.
        num_worker (int): Number of worker threads for data loading.
        dataset (UEA): UEA dataset to be used.
        scaler (Scaler): Scaler to normalize the data.
        window (int): Window size for the time series data. If not enough data, zeros will be used for padding.
        shuffle_train (bool): Whether to shuffle the training data.
    """
    def __init__(
        self,
        dataset: UEA,
        scaler: Scaler,
        window: int = 168,
        shuffle_train=True,
        batch_size: int = 32,
        num_worker: int = 3,
    ) -> None:

        self.batch_size = batch_size
        self.num_worker = num_worker
        self.dataset = dataset

        self.scaler = scaler
        self.window = window
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
        self.train_dataset = UEADataset(self.dataset, self.scaler, 'train', self.window, scaler_fit=True)
        self.val_dataset = UEADataset(self.dataset, self.scaler, 'test', self.window, scaler_fit=False)
        self.test_dataset = UEADataset(self.dataset, self.scaler, 'test', self.window, scaler_fit=False)

    def _load_dataloader(self):
        self.train_size = len(self.train_dataset)
        self.val_size = len(self.val_dataset)
        self.test_size = len(self.test_dataset)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_worker,
            drop_last=False,
            collate_fn=lambda x: collate_fn(x, max_len=self.window)
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
            drop_last=False,
            collate_fn=lambda x: collate_fn(x, max_len=self.window)
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
            drop_last=False,
            collate_fn=lambda x: collate_fn(x, max_len=self.window)
        )







def collate_fn(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """
    batch_size = len(data)
    features, raw_features, labels = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)

    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    scaled_X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        scaled_X[i, :end, :] = features[i][:end, :]
        X[i, :end, :] = raw_features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return scaled_X, X, targets, padding_masks



def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
