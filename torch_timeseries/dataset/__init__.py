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


synthetic_datasets = [
    'Dummy',
    'DummyGraph'
]

__all__ = forecast_datasets + classify_datasets + anomaly_datasets + synthetic_datasets