from torch_timeseries.scaler import StandardScaler
from torch_timeseries.dataset import  ETTh1
from torch_timeseries.dataloader import SlidingWindowTS, SlidingWindowTimeIndex

def test_single_variate_dataloader():
    dataloader = SlidingWindowTS(
        ETTh1(),
        batch_size=32,
        window=96,
        scaler=StandardScaler(),
        steps=12,
        freq='h',
        single_variate=True
    )
    
    for x, y, origin_x, origin_y, x_enc_date, y_enc_date in dataloader.train_loader:
        assert x.shape[1] == 96 and x.shape[2] == 1
        assert y.shape[1] == 12 and y.shape[2] == 1
        assert x_enc_date.shape[1] == 96 and x_enc_date.shape[2] == 4
        assert y_enc_date.shape[1] == 12 and y_enc_date.shape[2] == 4
        continue
    
    for x, y, origin_x, origin_y, x_enc_date, y_enc_date in dataloader.val_loader:
        continue
    
    for x, y, origin_x, origin_y, x_enc_date, y_enc_date in dataloader.test_loader:
        continue
    
    
    

def test_sliding_timeindex_dataloader():
    dataloader = SlidingWindowTimeIndex(
        ETTh1(),
        batch_size=32,
        window=96,
        scaler=StandardScaler(),
        steps=12,
        freq='h',
        num_worker=1,
    )
    
    for x, y, origin_x, origin_y, x_enc_date, y_enc_date, x_index, y_index in dataloader.train_loader:
        continue
    
    for x, y, origin_x, origin_y, x_enc_date, y_enc_date in dataloader.val_loader:
        continue
    
    for x, y, origin_x, origin_y, x_enc_date, y_enc_date in dataloader.test_loader:
        continue
    
def test_multi_variate_dataloader():
    dataloader = SlidingWindowTS(
        ETTh1(),
        batch_size=32,
        window=96,
        scaler=StandardScaler(),
        steps=12,
        freq='h',
    )
    
    for x, y, origin_x, origin_y, x_enc_date, y_enc_date in dataloader.train_loader:
        assert x.shape[1] == 96 and x.shape[2] == 7
        assert y.shape[1] == 12 and y.shape[2] == 7
        assert x_enc_date.shape[1] == 96 and x_enc_date.shape[2] == 4
        assert y_enc_date.shape[1] == 12 and y_enc_date.shape[2] == 4
        continue
    
    for x, y, origin_x, origin_y, x_enc_date, y_enc_date in dataloader.val_loader:
        continue
    
    for x, y, origin_x, origin_y, x_enc_date, y_enc_date in dataloader.test_loader:
        continue