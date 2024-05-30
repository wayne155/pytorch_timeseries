from sktime.datasets import load_from_tsfile_to_dataframe


import os
import resource
from ..core.dataset.dataset import Dataset, TimeSeriesDataset
from typing import Any, Callable, List, Optional
import torch
from torchvision.datasets.utils import (
    download_url,
    download_and_extract_archive,
    check_integrity,
)
import pandas as pd
import numpy as np
import torch.utils.data

def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


class UEA(TimeSeriesDataset):
    """
    Dataset class for datasets included in:
    Time Series Classification Archive (www.timeseriesclassification.com).
    
    For a list of available datasets, see https://www.timeseriesclassification.com/dataset.php.

    Attributes:
        train_df (pd.DataFrame): DataFrame containing the training data.
        train_labels (pd.DataFrame): DataFrame containing the labels for the training data.
        test_df (pd.DataFrame): DataFrame containing the test data.
        test_labels (pd.DataFrame): DataFrame containing the labels for the test data.
        num_classes (int): Number of classes in the dataset.
    """
    
    train_df : pd.DataFrame
    train_labels  : pd.DataFrame
    test_df : pd.DataFrame
    test_labels  : pd.DataFrame
    num_classes  : int

    def __init__(self, name, root='./data'):
        self.name = name
        self.root = root
        self.dir = os.path.join(root, self.name)
        os.makedirs(self.dir, exist_ok=True)
        
        self.download()
        
        self._load()
        
        self.num_features = self.train_features_data.shape[1]
        
    def download(self):
        download_and_extract_archive(
            f"https://www.timeseriesclassification.com/aeon-toolkit/{self.name}.zip",
            self.dir,
            filename=f"{self.name}.zip",
        )


    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df
    
    
    def _load(self) -> np.ndarray:
        self.train_filepath = os.path.join(self.dir, f"{self.name}_TRAIN.ts")
        self.test_filepath = os.path.join(self.dir, f"{self.name}_TEST.ts")

        self.train_df, self.train_labels = self.load_single(self.train_filepath)
        self.test_df, self.test_labels = self.load_single(self.test_filepath)
        
        self.train_features_data, self.train_labels_data = self.train_df.values, self.train_labels.values
        self.test_features_data, self.test_labels_data = self.test_df.values, self.test_labels.values

        self.num_classes = len(self.class_names)
                
        
        
        
        
