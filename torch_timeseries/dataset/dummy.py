

import numpy as np
import pandas as pd
from torch_timeseries.dataset.dataset import Freq, TimeSeriesDataset, TimeSeriesStaticGraphDataset

class Dummy(TimeSeriesDataset):
    name: str = 'dummy'
    num_features:int = 8
    sample_rate:int = 1
    length : int= 1000
    def download(self): 
        pass
    
    def _load(self):
        dates = pd.date_range(start='2022-01-01', end='2022-01-02', freq='t')

        data = np.random.rand(len(dates), 2)

        result = np.concatenate([dates[:, np.newaxis], data], axis=1)

        self.df = pd.DataFrame(result, columns=['date', 'data1', 'data2'])
        self.data = self.df.drop('date').values        
        return self.data


class DummyDatasetGraph(TimeSeriesStaticGraphDataset):
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


class DummyWithTime(TimeSeriesDataset):
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
    
    
    
    
class DummyWithTime(TimeSeriesDataset):
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






class DummyContinuous(TimeSeriesDataset):
    name: str = 'dummy'
    num_features: int = 10
    freq: Freq = Freq.minutes
    length: int = 10000

    def download(self):
        pass
    
    def _load(self):
        dates = pd.date_range(start='2022-01-01', periods=self.length, freq='t')
        # dates = pd.date_range(start='2022-01-01', end='2022-01-03', freq='t')
        data = np.zeros((len(dates), self.num_features))
        data[:3, :] = np.random.rand(3, self.num_features)  
        for i in range(3, len(dates)):
            for j in range(self.num_features):  
                data[i, j] = (data[i-1, j]+ data[i-2, j])/data[i-3, j] + np.sqrt(i)
        
        self.df = pd.DataFrame(data, columns=[ f"data{i}" for i in range(self.num_features)])
        self.df['date'] = dates
        self.dates = pd.DataFrame({'date': self.df.date})
        self.data = self.df.drop('date', axis=1).values
        
        return self.data
