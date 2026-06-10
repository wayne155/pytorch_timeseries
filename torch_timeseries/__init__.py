try:
    import torchmetrics as _tm  # noqa: F401 — must load before torch_timeseries.model
    del _tm                     # avoids libstdc++ conflict with matplotlib on some systems
except ImportError:
    pass

import torch_timeseries.dataset
import torch_timeseries.dataloader
import torch_timeseries.model
from .cli.exp import exp
from .results import RunResult, LocalBackend, WandbBackend
from .experiment import Experiment, register_model
