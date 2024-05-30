# https://github.com/Mcompetitions/M4-methods/tree/master/Dataset


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


class M4(TimeSeriesDataset):
    """
    The M4 forecasting competition, the continuation of the previous three ones organized by Spyros Makridakis (https://en.wikipedia.org/wiki/Makridakis_Competitions).
    collected from https://github.com/Mcompetitions/M4-methods/tree/master,we concate train,test files as the dataset.

    Parameters:
    category (str): The category of the dataset to load. Can be one of ['Daily', 'Hourly', 'Monthly', 'Quarterly', 'Weekly', 'Yearly']. Default is 'Daily'.
    root (str): The root directory to store the dataset. Default is './data'.

    Attributes:
    dir (str): The directory where the dataset is stored.
    dates (pd.DataFrame): The dates associated with the dataset.
    df (pd.DataFrame): The combined training and test data.
    data (np.ndarray): The time series data.

    Methods:
    download(): Download the dataset from the M4 repository.
    _load(): Load the dataset from the local files.
    """


    name: str = "M4"

    def __init__(self, root: str = "./data", category="Daily"):
        self.category = category
        self.root = root
        self.dir = os.path.join(root, self.name)
        os.makedirs(self.dir, exist_ok=True)

        self.download()
        self._process()
        self._load()

        self.dates: Optional[pd.DataFrame]

    def _download_category(self, category):
        download_url(
            f"https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/{category}-train.csv",
            self.dir,
            filename=f"{category}-train.csv",
        )

        download_url(
            f"https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/{category}-test.csv",
            self.dir,
            filename=f"{category}-test.csv",
        )
    def download(self):
        self._download_category(self.category)

    def _load(self) -> np.ndarray:

        self.train_file_path = os.path.join(self.dir, f"{self.category}-train.csv")
        self.test_file_path = os.path.join(self.dir, f"{self.category}-test.csv")

        self.tarin_df = pd.read_csv(self.train_file_path)
        self.test_df = pd.read_csv(self.test_file_path)
        self.df = pd.concat([self.tarin_df, self.test_df]).dropna(axis=1)
        self.data = self.df.iloc[:, 1:].to_numpy()
        return self.data
