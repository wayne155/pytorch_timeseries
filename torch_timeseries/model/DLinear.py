import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_timeseries.nn.kernels import MovingAvg
from torch_timeseries.nn.decomp import SeriesDecomp
import numpy as np




class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, seq_len, pred_len, enc_in, individual:bool = False, output_prob=0):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_prob = output_prob

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = SeriesDecomp(kernel_size)
        self.individual = individual
        self.channels = enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        if self.output_prob > 0:
            self.act = F.gelu
            self.projection = nn.Linear(
                enc_in * pred_len, self.output_prob)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        output= x.permute(0,2,1) # to [Batch, Output length, Channel]
        if self.output_prob > 0:
            output = output.reshape(output.shape[0], -1)
            # (batch_size, num_classes)
            output = self.projection(output)
        return output


    # def forecast(self, x_enc):
    #     return self(x_enc)

    # def imputation(self, x_enc):
    #     return self(x_enc)

    # def anomaly_detection(self, x_enc):
    #     return self(x_enc)

    # def classification(self, x_enc):
    #     # Encoder
    #     enc_out = self(x_enc)
    #     # Output
    #     # (batch_size, seq_length * d_model)
    #     output = enc_out.reshape(enc_out.shape[0], -1)
    #     # (batch_size, num_classes)
    #     output = self.projection(output)
    #     return output
