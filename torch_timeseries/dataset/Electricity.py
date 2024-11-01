import os
import resource
from ..core.dataset.dataset import Dataset, TimeSeriesDataset
from typing import Any, Callable, List, Optional
import torch
from torchvision.datasets.utils import download_and_extract_archive, check_integrity, download_url
import pandas as pd
import numpy as np
import torch.utils.data



class Electricity(TimeSeriesDataset):
    """
    The raw dataset is available at https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014.
    It contains the electricity consumption in kWh recorded every 15 minutes from 2011 to 2014.
    Some dimensions were equal to 0, so we eliminated the records from 2011.
    The final dataset contains electricity consumption data for 321 clients from 2012 to 2014.
    The data was converted to reflect hourly consumption.

    Attributes:
        name (str): Name of the dataset.
        num_features (int): Number of features in the dataset.
        freq (str): Frequency of the data points, in minutes.
        length (int): Length of the dataset.

    Methods:
        download():
            Downloads and extracts the dataset.
        _load():
            Loads the dataset into a NumPy array.
    """

    name:str= 'electricity'
    num_features: int = 321
    freq: str = 'yt' # in minutes
    length :int = 26304
    
    def download(self):
        download_url(
            "https://raw.githubusercontent.com/wayne155/multivariate_timeseries_datasets/main/electricity/electricity.csv",
            self.dir,
            filename="electricity.csv",
            md5="a1973ba4f4bed84136013ffa1ca27dc8",
        )
        
    def _load(self) -> np.ndarray:
        self.file_name = os.path.join(self.dir, 'electricity.csv')
        self.df = pd.read_csv(self.file_name, parse_dates=['date'])
        self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.drop("date", axis=1).values
        return self.data
    