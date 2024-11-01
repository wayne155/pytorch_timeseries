import os
import resource
from ..core.dataset.dataset import Dataset, TimeSeriesDataset, Freq
from typing import Any, Callable, List, Optional
import torch
from torchvision.datasets.utils import download_url, download_and_extract_archive, check_integrity
import pandas as pd
import numpy as np
import torch.utils.data



class ETTm1(TimeSeriesDataset):
    """
    Electricity Transformer Temperature minutely~(15 minutes level). This dataset contains 2-year data from two separate counties in China. Each data point consists of the target value "oil temperature" and 6 power load features.
    
    This dataset is collected by Informer and is available at https://github.com/zhouhaoyi/ETDataset.

    Attributes:
        name (str): Name of the dataset.
        freq (str): Frequency of the data points.
        num_features (int): Number of features in the dataset.
        length (int): Length of the dataset.

    Methods:
        download():
            Downloads and extracts the dataset.
        _load():
            Loads the dataset into a NumPy array.
        _process():
            Processes the dataset using the parent class's process method.
    """
    name:str= 'ETTm1'
    num_features: int = 7
    freq : Freq = 'yt'
    length : int  = 69680
    
    def download(self):
        download_url(
         "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
         self.dir,
         filename='ETTm1.csv',
         md5="82d6bd89109c63d075d99c1077b33f38"
        )

        
    def _load(self) -> np.ndarray:
        self.file_path = os.path.join(self.dir, 'ETTm1.csv')
        self.df = pd.read_csv(self.file_path,parse_dates=[0])
        self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.iloc[:, 1:].to_numpy()
        return self.data
    