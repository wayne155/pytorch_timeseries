import numpy as np
import pandas as pd
from torchvision.datasets.utils import download_and_extract_archive, check_integrity, download_url
import torch
from typing import Any, Callable, List, Optional
import os
import resource
from ..core.dataset.dataset import Dataset, TimeSeriesDataset


class ExchangeRate(TimeSeriesDataset):
    """
    The collection of the daily exchange rates of eight foreign countries including Australia, British, Canada, Switzerland, China, Japan, New Zealand, and Singapore ranging from 1990 to 2016.
    
    The raw data is collected from https://github.com/laiguokun/multivariate-time-series-data.

    Attributes:
        name (str): Name of the dataset.
        num_features (int): Number of features in the dataset.
        freq (str): Frequency of the data points (daily).
        length (int): Length of the dataset.

    Methods:
        download():
            Downloads and extracts the dataset.
        _load():
            Loads the dataset into a NumPy array.
    """

    name: str = 'ExchangeRate'
    num_features: int = 8
    freq: str = 'yd' # daily data
    length : int = 7588
    
    
    def download(self) -> None:
        download_url(
            "https://raw.githubusercontent.com/wayne155/multivariate_timeseries_datasets/main/exchange_rate/exchange_rate.csv",
            self.dir,
            filename="exchange_rate.csv",
            md5="2fc11972378a4c8817c1adfdde522bf9",
        )

    def _load(self) -> np.ndarray:
        self.file_name = os.path.join(self.dir, 'exchange_rate.csv')
        self.df = pd.read_csv(self.file_name, parse_dates=['date'])
        self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.drop("date", axis=1).values
        return self.data
