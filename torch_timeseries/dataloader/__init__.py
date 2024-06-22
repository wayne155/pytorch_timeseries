from ..scaler import StandardScaler, Scaler, MaxAbsScaler
from .sliding_window import SlidingWindow
from .sliding_window_ts import SlidingWindowTS
from .maskts import MaskTimeFeatureSet, MaskTS
from .uea import UEAClassification
from .anomaly import AnomalyLoader


forecast_loaders = [
    "SlidingWindowTS",
    "SlidingWindow",
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
