import os
import resource
from ..core.dataset import AnomalyDataset
from typing import Callable, List, Optional
import torch
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import pandas as pd
import numpy as np

class PSM(AnomalyDataset):
    """
    PSM is a public dataset from eBay's internal server nodes. It has 26 dimensions and contains performance indicators and system metrics from multiple servers, such as CPU usage, memory usage, network traffic, disk IO, etc.
    """
    
    name:str= 'PSM'
    
    def download(self) -> None:
        download_and_extract_archive(
            f"https://drive.usercontent.google.com/download?id=14gCVQRciS2hs2SAjXpqioxE4CUzaYkhb&confirm=t",
            self.dir,
            filename=f"PSM.zip",
        )

    def _load(self) -> np.ndarray:

        self.train_filepath = os.path.join(os.path.join(self.dir, "PSM"), 'train.csv')
        self.test_filepath = os.path.join(os.path.join(self.dir, "PSM"), 'test.csv')
        self.labels_filepath = os.path.join(os.path.join(self.dir, "PSM"), 'test_label.csv')
        
        self.train_data = np.nan_to_num(pd.read_csv(self.train_filepath).values[:, 1:])
        self.test_data = np.nan_to_num(pd.read_csv(self.test_filepath).values[:, 1:])
        self.test_labels = pd.read_csv(os.path.join(self.labels_filepath)).values[:, 1:]
        
        self.num_features = self.train_data.shape[1]
        

