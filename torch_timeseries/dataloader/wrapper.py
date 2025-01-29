from typing import Optional, TypeVar, Tuple

import pandas as pd
from ..scaler import Scaler

from torch_timeseries.utils.timefeatures import time_features
from torch_timeseries.core import TimeSeriesDataset, TimeseriesSubset
from torch.utils.data import Dataset
import numpy as np
import torch


class MultiStepTimeFeatureSet(Dataset):
    def __init__(self, dataset: TimeseriesSubset, scaler: Scaler, time_enc=0, window: int = 168, horizon: int = 3, steps: int = 2, single_variate=False, freq=None, scaler_fit=True):
        self.dataset = dataset
        self.window = window
        self.horizon = horizon
        self.steps = steps
        self.time_enc = time_enc
        self.scaler = scaler
        self.single_variate = single_variate
        
        self.num_features = self.dataset.num_features
        self.length = self.dataset.length
        if freq is None:
            self.freq = self.dataset.freq
        else:
            self.freq = freq
        if scaler_fit:
            self.scaler.fit(self.dataset.data)
        self.scaled_data = self.scaler.transform(self.dataset.data)
        self.date_enc_data = time_features(
            self.dataset.dates, self.time_enc, self.freq)
        assert len(self.dataset) - self.window - self.horizon + 1 - self.steps + 1 > 0, "Dataset is not long enough!!!"


    def transform(self, values):
        return self.scaler.transform(values)

    def inverse_transform(self, values):
        return self.scaler.inverse_transform(values)

    def __getitem__(self, index):
        # scaled_x : (B, T, N)
        # scaled_y : (B, O, N)
        # x : (B, T, N)
        # y : (B, O, N)
        # x_date_enc : (B, T, D)
        # y_date_eDc : (B, O, D)
        if isinstance(index, int):
            total_len  =  len(self.dataset) - self.window - self.horizon + 1 - self.steps + 1
            fea_dim= index // total_len
            index = index % total_len
            if self.single_variate:
                scaled_x = self.scaled_data[index:index+self.window, fea_dim:fea_dim+1]

                scaled_y = self.scaled_data[self.window + self.horizon - 1 +
                                    index:self.window + self.horizon - 1 + index+self.steps, fea_dim:fea_dim+1]
                
                x = self.dataset.data[index:index+self.window, fea_dim:fea_dim+1]
                
                y = self.dataset.data[self.window + self.horizon - 1 +
                                    index:self.window + self.horizon - 1 + index+self.steps, fea_dim:fea_dim+1]

                x_date_enc = self.date_enc_data[index:index+self.window]

                y_date_enc = self.date_enc_data[self.window + self.horizon -
                                                1 + index:self.window + self.horizon - 1 + index+self.steps]
            
            else:
                scaled_x = self.scaled_data[index:index+self.window]

                scaled_y = self.scaled_data[self.window + self.horizon - 1 +
                                    index:self.window + self.horizon - 1 + index+self.steps]
                
                x = self.dataset.data[index:index+self.window]
                
                y = self.dataset.data[self.window + self.horizon - 1 +
                                    index:self.window + self.horizon - 1 + index+self.steps]

                x_date_enc = self.date_enc_data[index:index+self.window]

                y_date_enc = self.date_enc_data[self.window + self.horizon -
                                                1 + index:self.window + self.horizon - 1 + index+self.steps]
            
            return scaled_x, scaled_y, x , y, x_date_enc, y_date_enc
        else:
            raise TypeError('Not surpported index type!!!')

    def __len__(self):
        if self.single_variate:
            return (len(self.dataset) - self.window - self.horizon + 1 - self.steps + 1)*self.num_features
        else:
            return len(self.dataset) - self.window - self.horizon + 1 - self.steps + 1




class MultivariateFast(Dataset):
    def __init__(self, dataset: TimeseriesSubset, scaler: Scaler = None, time_enc=0, window: int = 168, horizon: int = 3, single_variate=False, steps: int = 2, freq=None, scaler_transform=True, scaler_fit=False):
        self.dataset = dataset
        self.window = window
        self.horizon = horizon
        self.steps = steps
        self.time_enc = time_enc
        self.scaler = scaler
        assert len(dataset) != 0, "Empty dataset!!!"
        
        self.num_features = self.dataset.num_features
        self.length = self.dataset.length
        if freq is None:
            self.freq = self.dataset.freq
        else:
            self.freq = freq
        if scaler_fit and scaler:
            self.scaler.fit(self.dataset.data)
        
        if scaler_transform:
            self.scaled_data = self.scaler.transform(self.dataset.data)
        else:
            self.scaled_data = self.dataset.data
        self.date_enc_data = time_features(
            self.dataset.dates, self.time_enc, self.freq)
        total_len = window + horizon + steps - 1
        self.total_len = total_len
        
        self.window_split_data = self.dataset.data[:(len(self.dataset.data)//(total_len)*(total_len)), :].reshape(-1, total_len, self.num_features)
        # self.window_split_data = np.concatenate([self.window_split_data, self.dataset.data[-(window+horizon+steps - 1):, :][np.newaxis, ...]], axis=0)
        
        self.window_split_scaled_data = self.scaled_data[:(len(self.scaled_data)//(total_len)*(total_len)), :].reshape(-1, total_len, self.num_features)
        # self.window_split_scaled_data = np.concatenate([self.window_split_scaled_data, self.scaled_data[-(window+horizon+steps - 1):, :][np.newaxis, ...]], axis=0)
        
        self.window_split_date = self.date_enc_data[:(len(self.scaled_data)//(total_len)*(total_len)), :].reshape(-1, total_len, self.date_enc_data.shape[-1])
        # self.window_split_date = np.concatenate([self.window_split_date, self.date_enc_data[-(window+horizon+steps - 1):, :][np.newaxis, ...]], axis=0)
        assert len(self.dataset) - self.window - self.horizon + 1 - self.steps + 1 > 0, "Dataset is not long enough!!!"


    def transform(self, values):
        return self.scaler.transform(values)

    def inverse_transform(self, values):
        return self.scaler.inverse_transform(values)

    # @staticmethod
    # def get_length(self, dataset):
    #     (len(self.dataset.data)//(total_len)*(total_len))
        
    
    def __getitem__(self, index):
        # scaled_x : (B, T, N)
        # scaled_y : (B, O, N)
        # x : (B, T, N)
        # y : (B, O, N)
        # x_date_enc : (B, T, D)
        # y_date_eDc : (B, O, D)
        if isinstance(index, int):
            scaled_x = self.window_split_scaled_data[index][:self.window]
            scaled_y = self.window_split_scaled_data[index][-(self.horizon + self.steps - 1):]

            x = self.window_split_data[index][:self.window]
            y = self.window_split_data[index][-(self.horizon + self.steps - 1):]

            x_date_enc = self.window_split_date[index][:self.window]
            y_date_enc = self.window_split_date[index][-(self.horizon + self.steps - 1):]
            
            return scaled_x, scaled_y, x , y, x_date_enc, y_date_enc
        else:
            raise TypeError('Not surpported index type!!!')

    def __len__(self):
        return len(self.window_split_data) 






class MultiStepSet(Dataset):
    def __init__(self, dataset: TimeseriesSubset, scaler: Scaler, window: int = 168, horizon: int = 3, steps: int = 2, scaler_fit=True):
        self.dataset = dataset
        self.window = window
        self.horizon = horizon
        self.steps = steps
        self.scaler = scaler
        
        self.num_features = self.dataset.num_features
        self.length = self.dataset.length
        if scaler_fit:
            self.scaler.fit(self.dataset.data)
        self.scaled_data = self.scaler.transform(self.dataset.data)
        assert len(self.dataset) - self.window - self.horizon + 1 - self.steps + 1 > 0, "Dataset is not long enough!!!"


    def transform(self, values):
        return self.scaler.transform(values)

    def inverse_transform(self, values):
        return self.scaler.inverse_transform(values)

    def __getitem__(self, index):
        # scaled_x : (B, T, N)
        # scaled_y : (B, O, N)
        # x : (B, T, N)
        # y : (B, O, N)
        if isinstance(index, int):
            scaled_x = self.scaled_data[index:index+self.window]

            scaled_y = self.scaled_data[self.window + self.horizon - 1 +
                                 index:self.window + self.horizon - 1 + index+self.steps]
            
            x = self.dataset.data[index:index+self.window]
            
            y = self.dataset.data[self.window + self.horizon - 1 +
                                 index:self.window + self.horizon - 1 + index+self.steps]

            return scaled_x, scaled_y, x , y
        else:
            raise TypeError('Not surpported index type!!!')

    def __len__(self):
        return len(self.dataset) - self.window - self.horizon + 1 - self.steps + 1







class SingleStepWrapper(Dataset):

    def __init__(self, dataset: TimeSeriesDataset, window: int = 168, horizon: int = 3):
        self.dataset = dataset
        self.window = window
        self.horizon = horizon

    def __getitem__(self, index):
        return self.dataset.data[index:index+self.window], self.dataset.data[self.window + self.horizon - 1 + index]

    def __len__(self):
        return len(self.dataset) - self.window - self.horizon + 1


class MultiStepWrapper(Dataset):
    def __init__(self, dataset: TimeSeriesDataset, window: int = 168, horizon: int = 3, steps: int = 2):
        self.dataset = dataset
        self.window = window
        self.horizon = horizon
        self.steps = steps

    def __getitem__(self, index):
        return self.dataset.data[index:index+self.window], self.dataset.data[self.window + self.horizon - 1 + index:self.window + self.horizon - 1 + index+self.steps]

    def __len__(self):
        return len(self.dataset) - self.window - self.horizon + 1 - self.steps + 1


class SingStepFlattenWrapper(Dataset):
    def __init__(self, dataset: TimeSeriesDataset, window: int = 168, horizon: int = 3):
        self.dataset = dataset
        self.window = window
        self.horizon = horizon

    def __getitem__(self, index):
        x = self.dataset.data[index:index+self.window]
        y = self.dataset.data[self.window + self.horizon - 1 + index]
        return x.flatten(), y

    def __len__(self):
        return len(self.dataset) - self.window - self.horizon + 1


class MultiStepFlattenWrapper(Dataset):
    def __init__(self, dataset: TimeSeriesDataset, window: int = 168, horizon: int = 3, steps: int = 2):
        self.dataset = dataset
        self.window = window
        self.horizon = horizon
        self.steps = steps

    def __getitem__(self, index):
        x = self.dataset.data[index:index+self.window]
        y = self.dataset.data[self.window + self.horizon - 1 +
                              index:self.window + self.horizon - 1 + index+self.steps]
        return x.flatten(), y.flatten()

    def __len__(self):
        return len(self.dataset) - self.window - self.horizon + 1 - self.steps + 1
