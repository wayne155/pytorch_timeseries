from torch_timeseries.experiments.forecast import ForecastExp
from torch_timeseries.experiments.anomaly_detection import AnomalyDetectionExp
from torch_timeseries.experiments.imputation import ImputationExp
from torch_timeseries.experiments.uea_classification import UEAClassificationExp
from torch_timeseries.model.AFTForecaster import AFTForecaster


class AFTForecasterForecast(ForecastExp):
    model_type = AFTForecaster


class AFTForecasterAnomalyDetection(AnomalyDetectionExp):
    model_type = AFTForecaster


class AFTForecasterImputation(ImputationExp):
    model_type = AFTForecaster


class AFTForecasterUEAClassification(UEAClassificationExp):
    model_type = AFTForecaster
