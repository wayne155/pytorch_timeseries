from dataclasses import dataclass
from typing import Optional, Union

from ..scaler import Scaler
from ..dataset import TimeSeriesDataset
from torch.optim import Optimizer
from dataclasses import dataclass, asdict, field
from torch.nn import Module

@dataclass
class BaseIrrelevant:
    data_path: str = "./data"
    device: str = "cuda:0"
    num_worker: int = 20
    save_dir: str = "./results"
    experiment_label: str = ""

@dataclass
class BaseRelevant:
    model_type : Union[str, Module] = ""
    dataset_type: Union[str, TimeSeriesDataset] = ""
    optm_type: Union[str, Optimizer] = "Adam"
    scaler_type:  Union[str, Scaler] = "StandardScaler"
    loss_func_type: Union[str, Module] = ""
    batch_size: int = 32
    lr: float = 0.001
    l2_weight_decay: float = 0.0005
    epochs: int = 20
    patience: int = 5
    max_grad_norm: float = 5.0
    invtrans_loss: bool = False
    
    