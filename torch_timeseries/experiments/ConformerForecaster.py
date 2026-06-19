from torch_timeseries.experiments.forecast import ForecastExp
from torch_timeseries.experiments.anomaly_detection import AnomalyDetectionExp
from torch_timeseries.experiments.imputation import ImputationExp
from torch_timeseries.experiments.uea_classification import UEAClassificationExp
from torch_timeseries.model.ConformerForecaster import ConformerForecaster


class ConformerForecasterForecast(ForecastExp):
    model_type = ConformerForecaster


class ConformerForecasterAnomalyDetection(AnomalyDetectionExp):
    model_type = ConformerForecaster


class ConformerForecasterImputation(ImputationExp):
    model_type = ConformerForecaster


class ConformerForecasterUEAClassification(UEAClassificationExp):
    model_type = ConformerForecaster
