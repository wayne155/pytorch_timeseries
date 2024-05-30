import os
import resource
from ..core.dataset.dataset import Dataset, TimeSeriesDataset
from typing import Any, Callable, List, Optional
import torch
from torchvision.datasets.utils import download_url, download_and_extract_archive, check_integrity
import pandas as pd
import numpy as np
import torch.utils.data


class Weather(TimeSeriesDataset):
    """
    This dataset contains local climatological data for nearly 1,600 in US locations, covering 4 years from 2010 to 2013. Data points are collected every hour. Each data point consists of the target value “wet bulb” and 11 climate features.
    The raw data can be acquired at https://www.ncei.noaa.gov/data/local-climatological-data/.

    Attributes:
        name (str): Name of the dataset.
        num_features (int): Number of features in the dataset.
        length (int): Length of the dataset.
        freq (str): Frequency of the data points.

    Methods:
        download():
            Downloads and extracts the dataset.
        _load():
            Loads the dataset into a NumPy array.
    """
    
    name:str= 'weather'
    num_features: int = 21
    length : int  = 52696
    freq:str=  'h'
    
    
    def download(self):
        download_and_extract_archive(
         "https://raw.githubusercontent.com/wayne155/multivariate_timeseries_datasets/main/weather.zip",
         self.dir,
         filename='weather.zip',
         md5="fe40dc5c7e1787ad8ae63e2cadb5abfe"
        )
        
        
        
    def _load(self) -> np.ndarray:
        self.file_path =  os.path.join( os.path.join(self.dir, 'weather')  , 'weather.csv') 
        self.df = pd.read_csv(self.file_path,parse_dates=[0])
        self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.iloc[:, 1:].to_numpy()
        return self.data
    