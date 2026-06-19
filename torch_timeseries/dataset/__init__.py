from .dummies import Dummy, DummyGraph
from .Traffic import Traffic
from .ExchangeRate import ExchangeRate
from .SolarEnergy import SolarEnergy
from .Electricity import Electricity
from .ETTh1 import ETTh1
from .ETTh2 import ETTh2
from .ETTm1 import ETTm1
from .ILI import ILI
from .Weather import Weather
from .ETTm2 import ETTm2
from .M4 import M4
from .UEA import UEA
from .SWaT import SWaT
from .SMD import SMD
from .MSL import MSL
from .PSM import PSM
from .SMAP import SMAP
from .SimFreqCF import SimFreqCF
from .SimFreq import SimFreq
from .Sine import Sine
from .Stocks import Stocks

forecast_datasets = [
    'Traffic',
    'ExchangeRate',
    'SolarEnergy',
    'Electricity',
    'ETTh1',
    'ETTh2',
    'ETTm1',
    'ETTm2',
    'Weather',
    'ILI',
    'M4',
]


classify_datasets = [
    'UEA'
] 


anomaly_datasets = [
    'SWaT',
    'SMD',
    'SMAP',
    'MSL',
    'PSM',
] 


generation_datasets = [
    'Sine',
    'Stocks',
]

synthetic_datasets = [
    'Dummy',
    'DummyGraph',
    'SimFreqCF',
    'SimFreq'
]

__all__ = forecast_datasets + classify_datasets + anomaly_datasets + synthetic_datasets + generation_datasets
from .custom import CSVDataset, build_dataset, DataFrameDataset, from_dataframe
__all__ += ['CSVDataset', 'build_dataset', 'DataFrameDataset', 'from_dataframe']


import numpy as np
from ..core.dataset.dataset import DEFAULT_DATA_ROOT


def list_datasets(task: str = "all") -> list:
    """Return a sorted list of available dataset names.

    Parameters
    ----------
    task:
        Filter by task.  One of ``"forecast"``, ``"classify"``,
        ``"anomaly"``, ``"generation"``, ``"synthetic"``, or ``"all"``
        (default).

    Returns
    -------
    list of str

    Examples
    --------
    >>> from torch_timeseries.dataset import list_datasets
    >>> list_datasets("forecast")
    ['Electricity', 'ETTh1', 'ETTh2', ...]
    """
    mapping = {
        "forecast": forecast_datasets,
        "classify": classify_datasets,
        "anomaly": anomaly_datasets,
        "generation": generation_datasets,
        "synthetic": synthetic_datasets,
    }
    if task == "all":
        combined = []
        for v in mapping.values():
            combined.extend(v)
        return sorted(set(combined))
    if task not in mapping:
        raise ValueError(
            f"Unknown task {task!r}. Choose from {sorted(mapping)} or 'all'."
        )
    return sorted(mapping[task])


def load_dataset(name: str, root: str = DEFAULT_DATA_ROOT) -> np.ndarray:
    """Load a built-in dataset and return its data as a NumPy array.

    The dataset is downloaded automatically on the first call (stored in
    *root*).  Subsequent calls reuse the cached version.

    Parameters
    ----------
    name:
        Dataset name, e.g. ``"ETTh1"``, ``"Electricity"``.  Use
        :func:`list_datasets` to see all available names.
    root:
        Directory where datasets are cached (default
        ``~/.torchtimeseries/data``).

    Returns
    -------
    np.ndarray
        Shape ``(N, C)`` — N timesteps, C channels.

    Examples
    --------
    >>> from torch_timeseries.dataset import load_dataset
    >>> X = load_dataset("ETTh1")
    >>> X.shape
    (17420, 7)
    """
    import torch_timeseries.dataset as _ds
    if not hasattr(_ds, name):
        available = list_datasets()
        raise ValueError(
            f"Unknown dataset {name!r}. Available: {available}"
        )
    cls = getattr(_ds, name)
    dataset = cls(root=root)
    return dataset.data.astype(np.float32)


__all__ += ['list_datasets', 'load_dataset']
