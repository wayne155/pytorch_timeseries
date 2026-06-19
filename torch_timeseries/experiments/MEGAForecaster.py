from torch_timeseries.experiments.forecast import ForecastExp
from torch_timeseries.experiments.anomaly_detection import AnomalyDetectionExp
from torch_timeseries.experiments.imputation import ImputationExp
from torch_timeseries.experiments.uea_classification import UEAClassificationExp
from torch_timeseries.model.MEGAForecaster import MEGAForecaster


class MEGAForecasterForecast(ForecastExp):
    model_type = MEGAForecaster


class MEGAForecasterAnomalyDetection(AnomalyDetectionExp):
    model_type = MEGAForecaster


class MEGAForecasterImputation(ImputationExp):
    model_type = MEGAForecaster


class MEGAForecasterUEAClassification(UEAClassificationExp):
    model_type = MEGAForecaster
