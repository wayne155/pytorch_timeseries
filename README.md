[pypi-image]: https://badge.fury.io/py/torch-timeseries.svg
[pypi-url]: https://pypi.python.org/pypi/torch-timeseries
[docs-image]: https://readthedocs.org/projects/pytorch-timeseries/badge/?version=latest
[docs-url]: https://pytorch-timeseries.readthedocs.io/en/latest/?badge=latest



<p align="center">
  <img width="90%" src="https://raw.githubusercontent.com/wayne155/pytorch_timeseries/main/docs/_static/img/logo_text.jpg?sanitize=true" />
</p>

[![PyPI Version][pypi-image]][pypi-url]
[![Docs Status][docs-image]][docs-url]






# pytorch_timeseries
An all in one deep learning library that boost your timeseries research.
[Check the documentation for more detail](https://pytorch-timeseries.readthedocs.io/en/latest/).

Compared to previous libraries, pytorch_timeseries is 
- dataset automatically downloaded
- easy to use and extend 
- clear documentation 
- highly customizable 
- ..........



## installation



```
pip install torch-timeseries
```

> ⚠️⚠️⚠️ **Warning: We only support python version >= 3.8+**

### addtional install
For running Graph Nerual Network based models, pytorch_geometric is also needed.

```python
pip install torch_geometric

# Optional dependencies
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```


# Quick Start

## 1 Forecasting

### 1.1 download dataset
The dataset will be downloaded **automatically!!!!**
```python
from torch_timeseries.dataset import ETTh1
from torch_timeseries.dataloader import StandardScaler, SlidingWindow, SlidingWindowTS
from torch_timeseries.model import DLinear
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
dataset = ETTh1('./data')
```

### 1.2 setup scaler/dataloader

Once you setup a dataloader and pass a scaler into this dataloader, the scaler will be fitted on the training set.


```python
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

```
After this, you can access the train/val/test loader by `dataloader.train_loader/val_loader/test_loader` 

### 1.3 training



```python
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
```

### 1.4 val/test

```python
# val
model.eval()
for scaled_x, scaled_y, x, y, x_date_enc, y_date_enc in dataloader.val_loader:
    ....your validation code here...

# test
model.eval()
for scaled_x, scaled_y, x, y, x_date_enc, y_date_enc in dataloader.test_loader:
    ....your test code here...
```





## 2 Imputation

### 1. download dataset
The dataset will be downloaded **automatically!!!!**
```python
from torch_timeseries.dataset import ETTh1
from torch_timeseries.dataloader import StandardScaler, SlidingWindow, SlidingWindowTS
from torch_timeseries.model import DLinear
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
dataset = ETTh1('./data')
```

### 2. setup scaler/dataloader

Once you setup a dataloader and pass a scaler into this dataloader, the scaler will be fitted on the training set.


```python
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

```
After this, you can access the train/val/test loader by `dataloader.train_loader/val_loader/test_loader` 

### 3. training



```python
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
```

### 4. val/test

```python
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



