from torch_timeseries.dataset import M4
from torch_timeseries.dataloader import StandardScaler, SlidingWindow, SlidingWindowTS
from torch_timeseries.model import DLinear
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
dataset = M4('./data','Daily')
