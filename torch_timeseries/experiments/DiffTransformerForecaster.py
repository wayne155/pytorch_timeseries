from torch_timeseries.experiments.forecast import ForecastExp
from torch_timeseries.experiments.anomaly_detection import AnomalyDetectionExp
from torch_timeseries.experiments.imputation import ImputationExp
from torch_timeseries.experiments.uea_classification import UEAClassificationExp
from torch_timeseries.model.DiffTransformerForecaster import DiffTransformerForecaster


class DiffTransformerForecasterForecast(ForecastExp):
    model_type = DiffTransformerForecaster


class DiffTransformerForecasterAnomalyDetection(AnomalyDetectionExp):
    model_type = DiffTransformerForecaster


class DiffTransformerForecasterImputation(ImputationExp):
    model_type = DiffTransformerForecaster


class DiffTransformerForecasterUEAClassification(UEAClassificationExp):
    model_type = DiffTransformerForecaster
