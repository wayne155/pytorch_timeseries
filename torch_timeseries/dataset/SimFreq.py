

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Any, Callable, Generic, NewType, Optional, Sequence, TypeVar, Union
from torch import Tensor
import torch.utils.data
import os
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
from abc import ABC, abstractmethod



from enum import Enum, unique

from torch_timeseries.core import TimeSeriesDataset, BaseIrrelevant, BaseRelevant

class SimFreq(TimeSeriesDataset):
    name: str = 'SimFreq'
    num_features:int = 1
    sample_rate:int = 1
    length : int= 10000
    freq: str = 't'
    
    def download(self): 
        pass
    

    def _load(self):
        # n = 400
        # Generating date series
        dates = pd.date_range(start='2022-01-01', periods=self.length, freq='t')
        
        # Creating a data matrix
        data = np.zeros((len(dates), self.num_features))
        freqs_mag = ([(1, 2)])
        # freqs = (12,48,64)
        T = [(64, 200)]
        
        
        Periods = np.linspace(T[0][0], T[0][1], len(dates))
        mags = np.linspace(freqs_mag[0][0], freqs_mag[0][1], len(dates))
        


        # freqs_continue = []
        # for i in range(self.num_features):
        #     freqs_ = []
        #     freqs_.append(np.linspace(freqs1[i][0],freqs1[i][1], num=int(self.length*0.7)))
        #     freqs_.append(np.linspace(freqs1[i][1], freqs1[i][2], num=int(self.length*0.2)))
        #     freqs_.append(np.linspace(freqs1[i][2], freqs1[i][3] , num=int(self.length) - int(self.length*0.7) - int(self.length*0.2) ) )
        #     freqs_continue.append(np.concatenate(freqs_))

        
        # seqs = []
        # for i in range(self.num_features):
        #     seqs_ = []
        #     seqs_.append(np.linspace(freqs_mag[i][0],freqs_mag[i][1], num=int(self.length*0.7)))
        #     seqs_.append(np.linspace(freqs_mag[i][1], freqs_mag[i][2], num=int(self.length*0.2)))
        #     seqs_.append(np.linspace(freqs_mag[i][2], freqs_mag[i][3] , num=int(self.length) - int(self.length*0.7) - int(self.length*0.2) ) )
        #     seqs.append( np.concatenate(seqs_))
        # seq1 = np.linspace(1, 3, num=int(self.length*0.7))
        # seq2 = np.linspace(3, 2.5, num=int(self.length*0.2))
        # seq3 = np.linspace(2.5, 4 , num=int(self.length) - len(seq2) - len(seq1) )
        # combined_seq = np.concatenate((seq1, seq2, seq3))
        # Ts = 10
        t = np.arange(0, len(dates))

        for i in range(0, self.num_features):
            x = 0
            
            freq_signals  = 0 
            # for j in range(0, i+1):
            w = 2*np.pi / Periods
            freq_signals += np.sin(w * t)
            x = mags*freq_signals
            # x += +  (t // n)
            data[:, i] = x
        self.wt = w * t
        self.t = t
        self.w = w
        self.Periods = Periods
        # Creating DataFrame with specified column names
        self.df = pd.DataFrame(data, columns=[ f"data{i}" for i in range(self.num_features)])
        self.df['date'] = dates
        self.dates =  pd.DataFrame({'date': dates})
        self.data = self.df.drop('date', axis=1).values        
        return self.data
