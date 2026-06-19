from torch_timeseries.experiments.forecast import ForecastExp
from torch_timeseries.experiments.anomaly_detection import AnomalyDetectionExp
from torch_timeseries.experiments.imputation import ImputationExp
from torch_timeseries.experiments.uea_classification import UEAClassificationExp
from torch_timeseries.model.LiquidNetForecaster import LiquidNetForecaster


class LiquidNetForecasterForecast(ForecastExp):
    model_type = LiquidNetForecaster


class LiquidNetForecasterAnomalyDetection(AnomalyDetectionExp):
    model_type = LiquidNetForecaster


class LiquidNetForecasterImputation(ImputationExp):
    model_type = LiquidNetForecaster


class LiquidNetForecasterUEAClassification(UEAClassificationExp):
    model_type = LiquidNetForecaster
