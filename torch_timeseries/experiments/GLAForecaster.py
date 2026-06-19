from torch_timeseries.experiments.forecast import ForecastExp
from torch_timeseries.experiments.anomaly_detection import AnomalyDetectionExp
from torch_timeseries.experiments.imputation import ImputationExp
from torch_timeseries.experiments.uea_classification import UEAClassificationExp
from torch_timeseries.model.GLAForecaster import GLAForecaster


class GLAForecasterForecast(ForecastExp):
    model_type = GLAForecaster


class GLAForecasterAnomalyDetection(AnomalyDetectionExp):
    model_type = GLAForecaster


class GLAForecasterImputation(ImputationExp):
    model_type = GLAForecaster


class GLAForecasterUEAClassification(UEAClassificationExp):
    model_type = GLAForecaster
