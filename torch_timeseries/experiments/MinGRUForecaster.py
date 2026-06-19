from torch_timeseries.experiments.forecast import ForecastExp
from torch_timeseries.experiments.anomaly_detection import AnomalyDetectionExp
from torch_timeseries.experiments.imputation import ImputationExp
from torch_timeseries.experiments.uea_classification import UEAClassificationExp
from torch_timeseries.model.MinGRUForecaster import MinGRUForecaster


class MinGRUForecasterForecast(ForecastExp):
    model_type = MinGRUForecaster


class MinGRUForecasterAnomalyDetection(AnomalyDetectionExp):
    model_type = MinGRUForecaster


class MinGRUForecasterImputation(ImputationExp):
    model_type = MinGRUForecaster


class MinGRUForecasterUEAClassification(UEAClassificationExp):
    model_type = MinGRUForecaster
