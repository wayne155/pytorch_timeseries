import os
import resource
from ..core.dataset.dataset import Dataset, TimeSeriesDataset
from typing import Callable, List, Optional
import torch
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import pandas as pd
import numpy as np

class SMD(TimeSeriesDataset):
    """
    SMD (Server Machine Dataset, Su et al. (2019)) is a 5-week dataset collected from a large Internet company with 38 dimensions.
    """

    name:str= 'SMD'
    
    def download(self) -> None:
        download_and_extract_archive(
            f"https://drive.usercontent.google.com/download?id=1BgjQ7_2uqRrZ789Pijtpid5xpLTniywu&confirm=t",
            self.dir,
            filename=f"SMD.zip",
        )

    def _load(self) -> np.ndarray:

        self.train_filepath = os.path.join(os.path.join(self.dir, "SMD"), 'SMD_train.npy')
        self.test_filepath = os.path.join(os.path.join(self.dir, "SMD"), 'SMD_test.npy')
        self.labels_filepath = os.path.join(os.path.join(self.dir, "SMD"), 'SMD_test_label.npy')
        
        self.train_data = np.load(self.train_filepath)
        self.test_data = np.load(self.test_filepath)
        self.test_labels = np.load(self.labels_filepath)
        


        self.num_features = self.train_data.shape[1]
