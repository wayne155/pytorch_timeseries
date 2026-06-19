from torch_timeseries.experiments.forecast import ForecastExp
from torch_timeseries.experiments.anomaly_detection import AnomalyDetectionExp
from torch_timeseries.experiments.imputation import ImputationExp
from torch_timeseries.experiments.uea_classification import UEAClassificationExp
from torch_timeseries.model.FastFormerForecaster import FastFormerForecaster


class FastFormerForecasterForecast(ForecastExp):
    model_type = FastFormerForecaster


class FastFormerForecasterAnomalyDetection(AnomalyDetectionExp):
    model_type = FastFormerForecaster


class FastFormerForecasterImputation(ImputationExp):
    model_type = FastFormerForecaster


class FastFormerForecasterUEAClassification(UEAClassificationExp):
    model_type = FastFormerForecaster
