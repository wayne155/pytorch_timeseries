from torch_timeseries.experiments.forecast import ForecastExp
from torch_timeseries.experiments.anomaly_detection import AnomalyDetectionExp
from torch_timeseries.experiments.imputation import ImputationExp
from torch_timeseries.experiments.uea_classification import UEAClassificationExp
from torch_timeseries.model.RWKVForecaster import RWKVForecaster


class RWKVForecasterForecast(ForecastExp):
    model_type = RWKVForecaster


class RWKVForecasterAnomalyDetection(AnomalyDetectionExp):
    model_type = RWKVForecaster


class RWKVForecasterImputation(ImputationExp):
    model_type = RWKVForecaster


class RWKVForecasterUEAClassification(UEAClassificationExp):
    model_type = RWKVForecaster
