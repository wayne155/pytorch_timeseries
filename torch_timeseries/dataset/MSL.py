import os
import resource
from ..core.dataset import AnomalyDataset
from typing import Callable, List, Optional
import torch
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import pandas as pd
import numpy as np

class MSL(AnomalyDataset):
    """
    MSL is a public dataset from NASA with 55 dimensions showing the status of sensor and actuator data from the Mars Rover.
    """

    name:str= 'MSL'
    
    def download(self) -> None:
        download_and_extract_archive(
            f"https://drive.usercontent.google.com/download?id=14STjpszyi6D0B7BUHZ1L4GLUkhhPXE0G&confirm=t",
            self.dir,
            filename=f"MSL.zip",
        )

    def _load(self) -> np.ndarray:
        self.train_filepath = os.path.join(os.path.join(self.dir, "MSL"), 'MSL_train.npy')
        self.test_filepath = os.path.join(os.path.join(self.dir, "MSL"), 'MSL_test.npy')
        self.labels_filepath = os.path.join(os.path.join(self.dir, "MSL"), 'MSL_test_label.npy')
        
        self.train_data = np.load(self.train_filepath)
        self.test_data = np.load(self.test_filepath)
        self.test_labels = np.load(self.labels_filepath)
        
        self.num_features = self.train_data.shape[1]
        
        

