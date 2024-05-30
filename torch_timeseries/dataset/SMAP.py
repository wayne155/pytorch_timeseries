import os
import resource
from ..core.dataset import AnomalyDataset
from typing import Callable, List, Optional
import torch
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import pandas as pd
import numpy as np

class SMAP(AnomalyDataset):
    """
    SMAP is a multidimensional time series dataset collected by NASA's Mars rover. SMAP has 25 dimensional features and contains soil samples and telemetry information used by the Mars rover.
    """

    name:str= 'SMAP'
    
    def download(self) -> None:
        download_and_extract_archive(
            f"https://drive.usercontent.google.com/download?id=1kxiTMOouw1p-yJMkb_Q_CGMjakVNtg3X&confirm=t",
            self.dir,
            filename=f"SMAP.zip",
        )

    def _load(self) -> np.ndarray:

        self.train_filepath = os.path.join(os.path.join(self.dir, "SMAP"), 'SMAP_train.npy')
        self.test_filepath = os.path.join(os.path.join(self.dir, "SMAP"), 'SMAP_test.npy')
        self.labels_filepath = os.path.join(os.path.join(self.dir, "SMAP"), 'SMAP_test_label.npy')
        
        self.train_data = np.load(self.train_filepath)
        self.test_data = np.load(self.test_filepath)
        self.test_labels = np.load(self.labels_filepath)
        
        

        self.num_features = self.train_data.shape[1]
