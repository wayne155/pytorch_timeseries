from torch_timeseries.experiments.forecast import ForecastExp
from torch_timeseries.experiments.anomaly_detection import AnomalyDetectionExp
from torch_timeseries.experiments.imputation import ImputationExp
from torch_timeseries.experiments.uea_classification import UEAClassificationExp
from torch_timeseries.model.xLSTMForecaster import xLSTMForecaster


class xLSTMForecasterForecast(ForecastExp):
    model_type = xLSTMForecaster


class xLSTMForecasterAnomalyDetection(AnomalyDetectionExp):
    model_type = xLSTMForecaster


class xLSTMForecasterImputation(ImputationExp):
    model_type = xLSTMForecaster


class xLSTMForecasterUEAClassification(UEAClassificationExp):
    model_type = xLSTMForecaster
