from torch_timeseries.experiments.forecast import ForecastExp
from torch_timeseries.experiments.anomaly_detection import AnomalyDetectionExp
from torch_timeseries.experiments.imputation import ImputationExp
from torch_timeseries.experiments.uea_classification import UEAClassificationExp
from torch_timeseries.model.NystromForecaster import NystromForecaster


class NystromForecasterForecast(ForecastExp):
    model_type = NystromForecaster


class NystromForecasterAnomalyDetection(AnomalyDetectionExp):
    model_type = NystromForecaster


class NystromForecasterImputation(ImputationExp):
    model_type = NystromForecaster


class NystromForecasterUEAClassification(UEAClassificationExp):
    model_type = NystromForecaster
