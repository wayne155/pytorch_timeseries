[TOC]
# pytorch_timeseries
An all in one deep learning library that boost your timeseries research.

# install
> Note:This library assumes that you've installed Pytorch according to it's official website, the basic dependencies of torch > > related libraries may not be listed in the requirements files:
https://pytorch.org/get-started/locally/


## requirements

### run time requirements

The recommended python version is 3.8.1+.
Please first install torch according to your environment.
```
pip3 install torch torchvision torchaudio
```

For running Graph Nerual Network based models, pytorch_geometric is also needed.

```

pip install torch_geometric

# Optional dependencies
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

>check your torch & cuda version before you execute the command above
>```
>python -c "import torch; print(torch.__version__)"
>python -c "import torch; print(torch.version.cuda)"
>```




### tests  requirements



### dev requirements


