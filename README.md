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
- install and run! 
- ..........



## 1. installation



```
pip install torch-timeseries
```

> ⚠️⚠️⚠️ **Warning: We only support python version >= 3.8+**


## 2. Running Implemented Experiments

### Forecast
```python
# running DLinear Forecast on dataset ETTh1 with seed = 3 
pytexp --model DLinear --task Forecast --dataset_type ETTh1 run 3
# running DLinear Forecast on dataset ETTh1 with seeds=[1,2,3]
pytexp --model DLinear --task Forecast --dataset_type ETTh1 runs '[1,2,3]'
```


### Imputation
```python
# running DLinear Imputation on dataset ETTh1 with seed = 3 
pytexp --model DLinear --task Imputation --dataset_type ETTh1 run 3
# running DLinear Imputation on dataset ETTh1 with seed = [1,2,3] 
pytexp --model DLinear --task Imputation --dataset_type ETTh1 runs '[1,2,3]'
```
### UEAClassification
```python
# running DLinear UEAClassification on dataset EthanolConcentration with seed = 3 
pytexp --model DLinear --task UEAClassification --dataset_type EthanolConcentration run 3
# running DLinear UEAClassification on dataset EthanolConcentration with seed = [1,2,3] 
pytexp --model DLinear --task UEAClassification --dataset_type EthanolConcentration runs '[1,2,3]'
```

### AnomalyDetection
```python
# running DLinear AnomalyDetection on dataset MSL with seed = [1,2,3] 
pytexp --model DLinear --task AnomalyDetection --dataset_type MSL run 3
# running DLinear AnomalyDetection on dataset MSL with seed = [1,2,3] 
pytexp --model DLinear --task AnomalyDetection --dataset_type MSL runs 3
```


# Development Milestones
## Implemented Datasets
Full list of datasets can be found at [Documentation](https://pytorch-timeseries.readthedocs.io/en/latest/modules/dataset.html).
| Datasets | Forecasting | Imputation | Anomaly | Classification|
| --------- | ------- | ------- | ------- | ------- |
| [ETTh1](https://ojs.aaai.org/index.php/AAAI/article/view/17325)   | ✅ |✅ |  |  |
| [ETTh2](https://ojs.aaai.org/index.php/AAAI/article/view/17325)   | ✅ |✅ |  |  |
| [ETTm1](https://ojs.aaai.org/index.php/AAAI/article/view/17325)   | ✅ |✅ |  |  |
| [ETTm2](https://ojs.aaai.org/index.php/AAAI/article/view/17325)   | ✅ |✅ |  |  |
| [......And More](https://pytorch-timeseries.readthedocs.io/en/latest/modules/dataset.html)   | ✅ |✅ | ✅ | ✅ |

## Implemented Tasks

- [x] Forecast
- [x] Classfication (for UEA datasets)
- [x] Anomaly Detection 
- [x] Imputation
- [ ] You can fill this check box! (contribute to develop your own task!)

## Implemented Models

| Models | Forecasting | Imputation | Anomaly | Classification|
| --------- | ------- | ------- | ------- | ------- |
| [Informer (2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17325)   | ✅ |✅ |✅ |✅ |
| [Autoformer (2021)](https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html)   | ✅ |✅ |✅ |✅ |
| [FEDformer (2022)](https://proceedings.mlr.press/v162/zhou22g.html)   | ✅ |✅ |✅ |✅ |
| [DLinear (2022)](https://ojs.aaai.org/index.php/AAAI/article/view/26317)   | ✅ |✅ |✅ |✅ |
| [PatchTST (2022)](https://openreview.net/forum?id=Jbdc0vTOcol&trk=public_post_comment-text)   | ✅ |✅ |✅ |✅ |
| [iTransformer (2024)](https://openreview.net/forum?id=JePfAI8fah)   | ✅ |✅ |✅ |✅ |

<!-- ## Implemented Datasets
Currently we have implemented all popular datasets, including

| Datasets | Forecasting | Imputation | Anomaly | Classification|
| --------- | ------- | ------- | ------- | ------- |
| [ETTh1](https://ojs.aaai.org/index.php/AAAI/article/view/26317)   | ✅ |✅ |  |  |
| [ETTh2](https://ojs.aaai.org/index.php/AAAI/article/view/26317)   | ✅ |✅ |  |  |
| [ETTm1](https://ojs.aaai.org/index.php/AAAI/article/view/26317)   | ✅ |✅ |  |  |
| [ETTm2](https://ojs.aaai.org/index.php/AAAI/article/view/26317)   | ✅ |✅ |  |  |

[Check the documentation for more detail](https://pytorch-timeseries.readthedocs.io/en/latest/).
  -->

#  Customizing Your Own Pipeline

we provide examples of :
- [forecast](https://github.com/wayne155/pytorch_timeseries/blob/main/examples/forecast.py)
- [imputation](https://github.com/wayne155/pytorch_timeseries/blob/main/examples/mask.py)
- [anomaly detection](https://github.com/wayne155/pytorch_timeseries/blob/main/examples/anomaly.py)
- [UEA classfication](https://github.com/wayne155/pytorch_timeseries/blob/main/examples/ueaclass.py)

Detail of customize forecasting pipeline is as follows:

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


# Dev Install 

## install requirements
> Note:This library assumes that you've installed Pytorch according to it's official website, the basic dependencies of torch > > related libraries may not be listed in the requirements files:
https://pytorch.org/get-started/locally/

**The recommended python version is 3.8.1+.**
1. fork this project 

2. clone this project (latest version)
```
git clone https://github.com/wayne155/pytorch_timeseries
```

3.  install requirements.
```
pip install -r ./requirements.txt
```

4. change some code and push to the forked repo

5. create a pull request to this repo
