

import numpy as np
import pandas as pd
from torch_timeseries.core import Freq, TimeSeriesDataset, TimeSeriesStaticGraphDataset


class DummyGraph(TimeSeriesStaticGraphDataset):
    """
    Dummy graph dataset for testing purposes.

    Attributes:
        name (str): Name of the dataset.
        num_features (int): Number of features in the dataset.
        freq (Freq): Frequency of the data points.
        length (int): Length of the dataset.

    Methods:
        _load_static_graph():
            Loads a static adjacency matrix for the graph.
        download():
            Placeholder method for downloading data.
        _load():
            Loads the dataset into a NumPy array.
    """


    name: str = 'dummy_graph'
    num_features:int = 5
    freq : Freq = Freq.minutes
    length : int = 1440
    
    def _load_static_graph(self):
        self.adj = np.ones((self.num_features, self.num_features))
    def download(self): 
        pass
    
    def _load(self):
        self._load_static_graph()
        dates = pd.date_range(start='2022-01-01',periods=self.length, freq='t')

        data = np.random.rand(len(dates), self.num_features)
        result = {'date': dates}
        # iterate to get above df
        for i in range(data.shape[1]):  
            key = f'data{i+1}' 
            result[key] = data[:, i] 
        self.df = pd.DataFrame(result)

        self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.drop('date', axis=1).values        
        return self.data

