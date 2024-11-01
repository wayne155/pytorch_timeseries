import os
import resource
from ..core.dataset.dataset import Dataset, TimeSeriesDataset
from typing import Any, Callable, List, Optional
import torch
from torchvision.datasets.utils import download_url, download_and_extract_archive, check_integrity
import pandas as pd
import numpy as np
import torch.utils.data



class ILI(TimeSeriesDataset):
    """
    ILI (Influenza-Like Illness) records the weekly ratio of influenza-like illness patients versus the total patients 
    by the Centers for Disease Control and Prevention (CDC) of the United States from 2002 to 2021.
    
    This dataset contains 966 data points, each with 7 features, and the data frequency is hourly ('h').

    Collected from: https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html.

    Attributes:
        name (str): Name of the dataset, it is 'ILI'.
        num_features (int): Number of features in the dataset, it is 7.
        length (int): Total length of the dataset, it is 966.
        freq (str): Frequency of the data points, it is 'h'.

    Args:
        TimeSeriesDataset (_type_): The base class for time series datasets.

    Returns:
        np.ndarray: The data loaded from the CSV file.

    Methods:
        download(): Downloads and extracts the dataset archive from the specified URL.
        _load() -> np.ndarray: Loads the data from the CSV file and returns it as a numpy array.
    """
    

    name:str= 'ILI'
    num_features: int = 7
    length : int  = 966
    freq:str = 'yh'
    
    def download(self):
        download_and_extract_archive(
         "https://raw.githubusercontent.com/wayne155/multivariate_timeseries_datasets/main/illness.zip",
         self.dir,
         filename='illness.zip',
         md5="e18542fafd509ad52ffedde1b0f1018d"
        )
        
        
    def _load(self) -> np.ndarray:
        self.file_path =os.path.join( os.path.join(self.dir, 'illness')  , 'national_illness.csv')
        self.df = pd.read_csv(self.file_path,parse_dates=[0])
        self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.iloc[:, 1:].to_numpy()
        return self.data
    