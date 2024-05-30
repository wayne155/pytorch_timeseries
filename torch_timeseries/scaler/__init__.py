from ..core.scaler import Scaler
from .maxabs import MaxAbsScaler
from .standard import StandardScaler


scalers = [
    'MaxAbsScaler',
    'StandardScaler',
]

__all__ = scalers