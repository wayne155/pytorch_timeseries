from torch_timeseries.experiments.forecast import ForecastExp
from torch_timeseries.experiments.anomaly_detection import AnomalyDetectionExp
from torch_timeseries.experiments.imputation import ImputationExp
from torch_timeseries.experiments.uea_classification import UEAClassificationExp
from torch_timeseries.model.EchoStateForecaster import EchoStateForecaster


class EchoStateForecasterForecast(ForecastExp):
    model_type = EchoStateForecaster


class EchoStateForecasterAnomalyDetection(AnomalyDetectionExp):
    model_type = EchoStateForecaster


class EchoStateForecasterImputation(ImputationExp):
    model_type = EchoStateForecaster


class EchoStateForecasterUEAClassification(UEAClassificationExp):
    model_type = EchoStateForecaster
