# pytorch_timeseries
An all in one deep learning library that boost your timeseries research.

## installation
```
pip install pytorch-timeseries
```

## documentation
See [Documentation](https://pytorch-timeseries.readthedocs.io/en/latest/).

# Quick Start

```python
from torch_timeseries.dataset import ETTh1
from torch_timeseries.dataloader import StandardScaler, SlidingWindow, SlidingWindowTS
from torch_timeseries.model import DLinear
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
dataset = ETTh1('./data')
scaler = StandardScaler()
dataloader = SlidingWindowTS(dataset, 
                        window=96,
                        horizon=1,
                        steps=336,
                        batch_size=32, 
                        train_ratio=0.7, 
                        val_ratio=0.2, 
                        scaler=scaler,
                        )


model = DLinear(dataloader.window, dataloader.steps, dataset.num_features, individual= True)

optimizer = Adam(model.parameters())
loss_function = MSELoss()


# train
model.train()
for scaled_x, scaled_y, x, y, x_date_enc, y_date_enc in dataloader.train_loader:
    optimizer.zero_grad()
    
    scaled_x = scaled_x.float()
    scaled_y = scaled_y.float()
    scaled_pred_y = model(scaled_x) 
    
    loss = loss_function(scaled_pred_y, scaled_y)
    loss.backward()
    optimizer.step()
    print(loss)
# val
model.eval()
for scaled_x, scaled_y, x, y, x_date_enc, y_date_enc in dataloader.val_loader:
    scaled_x = scaled_x.float()
    scaled_y = scaled_y.float()
    scaled_pred_y = model(scaled_x) 
    loss = loss_function(scaled_pred_y, scaled_y)
    

# test
model.eval()
for scaled_x, scaled_y, x, y, x_date_enc, y_date_enc in dataloader.test_loader:
    scaled_x = scaled_x.float()
    scaled_y = scaled_y.float()
    scaled_pred_y = model(scaled_x) 
    loss = loss_function(scaled_pred_y, scaled_y)
    

```



# dev install 

# install requirements
> Note:This library assumes that you've installed Pytorch according to it's official website, the basic dependencies of torch > > related libraries may not be listed in the requirements files:
https://pytorch.org/get-started/locally/

**The recommended python version is 3.8.1+.**
Please first install torch according to your environment.
```
pip3 install torch torchvision torchaudio
```

For running Graph Nerual Network based models, pytorch_geometric is also needed.

```python
pip install torch_geometric

# Optional dependencies
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

>check your torch & cuda version before you execute the command above
>```python
>python -c "import torch; print(torch.__version__)"
>python -c "import torch; print(torch.version.cuda)"
>```
