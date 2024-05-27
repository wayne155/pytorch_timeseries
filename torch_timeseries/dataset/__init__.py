from .dummies import Dummy, DummyGraph
from .dataset import TimeSeriesDataset, TimeseriesSubset
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
from .sp500 import SP500
from .M4 import M4
from .EthanolConcentration import EthanolConcentration
from .FaceDetection import FaceDetection
from .Handwriting import Handwriting
from .Heartbeat import Heartbeat
from .JapaneseVowels import JapaneseVowels
from .PEMS_SF import PEMS_SF
from .SelfRegulationSCP1 import SelfRegulationSCP1
from .SelfRegulationSCP2 import SelfRegulationSCP2
from .SpokenArabicDigits import SpokenArabicDigits
from .EthanolConcentration import EthanolConcentration
from .UWaveGestureLibrary import UWaveGestureLibrary
from .UEA import UEA


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
    'SP500',
    'M4',
]


classify_datasets = [
    'EthanolConcentration',
    'FaceDetection',
    'Handwriting',
    'Heartbeat',
    'JapaneseVowels',
    'PEMS_SF',
    'SelfRegulationSCP1',
    'SelfRegulationSCP2',
    'SpokenArabicDigits',
    'UWaveGestureLibrary',
    'UEA'
] 


anomaly_datasets = [] 


synthetic_datasets = [
    'Dummy',
    'DummyGraph'
]

__all__ = forecast_datasets + classify_datasets + anomaly_datasets + synthetic_datasets
forecast_datasets
classify_datasets