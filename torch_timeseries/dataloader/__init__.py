from ..scaler import StandardScaler, Scaler, MaxAbsScaler
from .sliding_window import SlidingWindow
from .sliding_window_ts import SlidingWindowTS
from .maskts import MaskTimeFeatureSet, MaskTS
from .uea import UEAClassification
from .ETT import ETTHLoader, ETTMLoader
from .anomaly import AnomalyLoader
from .wrapper import MultivariateFast
from .noverlap_window_ts import NoneOverlapWindowTS

forecast_loaders = [
    "SlidingWindowTS",
    "SlidingWindow",
    "NoneOverlapWindowTS",
]

imputation_loaders = [
    "MaskTS",
    "MaskTimeFeatureSet",
]


classification_loaders = [
    "UEAClassification",
]

anomaly_loaders = [
    "AnomalyLoader",
]

scalers = [
    "StandardScaler",
    "MaxAbsScaler",
    "MinMaxScaler",
]

data_loaders = (
    forecast_loaders + anomaly_loaders + classification_loaders + imputation_loaders
)
__all__ = data_loaders + scalers
