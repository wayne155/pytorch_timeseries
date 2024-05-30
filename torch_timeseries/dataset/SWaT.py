import os
import resource
from ..core.dataset.dataset import Dataset, TimeSeriesDataset
from typing import Callable, List, Optional
import torch
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import pandas as pd
import numpy as np

class SWaT(TimeSeriesDataset):
    """
    SWaT (Safe Water Treatment, Mathur & Tippenhauer (2016)) is obtained from 51 sensors in continuously operating critical infrastructure systems.
    """

    name:str= 'SWaT'
    
    def download(self) -> None:
        download_and_extract_archive(
            f"https://drive.usercontent.google.com/download?id=1eRKQwJhqmUD4LkWnqNy1cdIz3W_y6EtW&confirm=t",
            self.dir,
            filename=f"SWaT.zip",
        )

    def _load(self) -> np.ndarray:
        self.train_filepath = os.path.join(os.path.join(self.dir, "SWaT"), 'swat_train2.csv')
        self.test_filepath = os.path.join(os.path.join(self.dir, "SWaT"), 'swat2.csv')
        
        self.train_df = pd.read_csv(self.train_filepath)
        self.test_df = pd.read_csv(self.test_filepath)
        
        self.train_data, self.train_labels = self.train_df.values[:, :-1], self.train_df.values[:, -1:]
        self.test_data, self.test_labels = self.test_df.values[:, :-1], self.test_df.values[:, -1:]
        
        self.num_features = self.train_data.shape[1]
        