

import numpy as np
import pandas as pd
from torch_timeseries.core import Freq, TimeSeriesDataset, TimeSeriesStaticGraphDataset


class Dummy(TimeSeriesDataset):
    """
    Dummy dataset for testing purposes.

    Attributes:
        name (str): Name of the dataset.
        num_features (int): Number of features in the dataset.
        freq (Freq): Frequency of the data points.
        length (int): Length of the dataset.

    Methods:
        download():
            Placeholder method for downloading data.
        _load():
            Loads the dataset into a NumPy array.
    """

    name: str = 'dummy'
    num_features:int = 2
    freq : Freq = Freq.minutes
    length : int = 1440
    def download(self): 
        pass
    
    def _load(self):
        dates = pd.date_range(start='2022-01-01', end='2022-01-02', freq='t')

        data = np.random.rand(len(dates), 2)
        self.df = pd.DataFrame({'date': dates, 'data1': data[:, 0],'data2': data[:, 1]})
        self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.drop('date', axis=1).values        
        return self.data
    
    
    