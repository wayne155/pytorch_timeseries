# https://github.com/Mcompetitions/M4-methods/tree/master/Dataset


import os
import resource
from .dataset import Dataset, TimeSeriesDataset
from typing import Any, Callable, List, Optional
import torch
from torchvision.datasets.utils import (
    download_url,
    download_and_extract_archive,
    check_integrity,
)
import pandas as pd
import numpy as np
import torch.utils.data


class PEMS_SF(TimeSeriesDataset):
    """

    """


    name: str = "PEMS-SF"

    def download(self):
        download_and_extract_archive(
            f"https://www.timeseriesclassification.com/aeon-toolkit/PEMS-SF.zip",
            self.dir,
            filename=f"PEMS-SF.zip",
        )



    def _load(self) -> np.ndarray:
        pass
