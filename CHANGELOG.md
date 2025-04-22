## 0.0.3
fixed: dependencies issue of torch-timeserie pypi package 


## 0.1.0

func: Forecast/Imputation/UEAClassification AnomalyDetection experiments
func: pytexp entrypoint


## 0.1.1
fix: fix a typo of the standard scaler
fix: fix some cli errors

## 0.1.2
func: add Informer Forecast/Imputation/AnomalyDetection/UEAClassification
func: add Autoformer Forecast/Imputation/AnomalyDetection/UEAClassification
func: add FEDformer Forecast/Imputation/AnomalyDetection/UEAClassification
func: add PatchTST Forecast/Imputation/AnomalyDetection/UEAClassification

## 0.1.3

func: add iTransformer Forecast/Imputation/AnomalyDetection/UEAClassification
fix: fix a typo of Informer d_layer config


## 0.1.6

func: adding a new model CATS
func: adding year into time features
func: add popular and set as default split, ETT for 6:2:2, others for 7:1:2
fix: default using only train data to scale 


## 0.1.7

func: we make the default dataloader settings identical with Time-Series-Libary



## 0.1.8

func: add new dataset wrapper, MultivariateFast, to split by window not by steps.

## 0.1.9

func: add a nonoverlap dataloader
fix: change default time encoding to 3, for stable data range in 0~1. We found that data like 30, 2024 will cause unstable training or even corrupted training process.


## 0.1.10

fix: fix data loading bugs


## 0.1.12

update: update iTransformer default configurations
update: update Forecast default parameter (l2_decay=0, lr=0.0001)
update: update timeenc config (timeenc=0)

## 0.1.13
bug fixed
## 0.1.14

update: change ident to seed+md5