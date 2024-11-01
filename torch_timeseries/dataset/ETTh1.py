import os
import resource
from ..core.dataset.dataset import Dataset, Freq, TimeSeriesDataset
from typing import Any, Callable, List, Optional
import torch
from torchvision.datasets.utils import download_url, download_and_extract_archive, check_integrity
import pandas as pd
import numpy as np
import torch.utils.data



class ETTh1(TimeSeriesDataset):
    """
    Electricity Transformer Temperature hourly. This dataset contains 2-year data from two separate counties in China. Each data point consists of the target value "oil temperature" and 6 power load features.
    
    This dataset is collected by Informer and is available at https://github.com/zhouhaoyi/ETDataset.

    Attributes:
        name (str): Name of the dataset.
        num_features (int): Number of features in the dataset.
        length (int): Length of the dataset.
        freq (str): Frequency of the data points.
        windows (int): Number of windows.

    Methods:
        download():
            Downloads and extracts the dataset.
        _load():
            Loads the dataset into a NumPy array.
    """
    
    name:str= 'ETTh1'
    num_features: int = 7
    length : int  = 17420
    freq : Freq = 'yh'
    
    def download(self):
        download_url(
         "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
         self.dir,
         filename='ETTh1.csv',
         md5="8381763947c85f4be6ac456c508460d6"
        )
        
    def _load(self) -> np.ndarray:
        self.file_path = os.path.join(self.dir, 'ETTh1.csv')
        self.df = pd.read_csv(self.file_path,parse_dates=[0])
        self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.iloc[:, 1:].to_numpy()
        return self.data
    
    
    