import os
import resource
from ..core.dataset.dataset import Dataset, TimeSeriesDataset
from typing import Callable, List, Optional
import torch
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import pandas as pd
import numpy as np

class Traffic(TimeSeriesDataset):
    """Traffic dataset for road occupancy rates in the San Francisco Bay area.

    The raw data is sourced from the California Department of Transportation
    and is available at http://pems.dot.ca.gov. This dataset contains 48 months
    (2015-2016) of hourly road occupancy rates (between 0 and 1) measured by
    different sensors on freeways in the San Francisco Bay area.

    Attributes:
        name (str): Name of the dataset.
        num_features (int): Number of features in the dataset.
        length (int): Length of the dataset.
        freq (str): Frequency of the data points.

    Methods:
        download(): Downloads and extracts the dataset.
        _load(): Loads the dataset into a NumPy array.
    """

    name:str= 'traffic'
    num_features: int = 862
    length : int = 17544
    freq:str = 'h'
    
    def download(self) -> None:
        download_and_extract_archive(
            "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt.gz",
            self.dir,
            filename="traffic.txt.gz",
            md5="db745d0c9f074159581a076cbb3f23d6",
        )
        
    def _load(self) -> np.ndarray:
        self.file_name = os.path.join(self.dir, 'traffic.txt')
        self.df = pd.read_csv(self.file_name, header=None)
        self.df['date'] = pd.date_range(start='01/01/2015 00:00', periods=self.length, freq='1H')  # '5T' for 5 minutes
        self.dates =  pd.DataFrame({'date': self.df['date'] })

        # self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.drop("date", axis=1).values
        # self.data = np.loadtxt(self.file_name, delimiter=',')
        return self.data
        