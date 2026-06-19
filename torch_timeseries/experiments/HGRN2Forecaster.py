from torch_timeseries.experiments.forecast import ForecastExp
from torch_timeseries.experiments.anomaly_detection import AnomalyDetectionExp
from torch_timeseries.experiments.imputation import ImputationExp
from torch_timeseries.experiments.uea_classification import UEAClassificationExp
from torch_timeseries.model.HGRN2Forecaster import HGRN2Forecaster


class HGRN2ForecasterForecast(ForecastExp):
    model_type = HGRN2Forecaster


class HGRN2ForecasterAnomalyDetection(AnomalyDetectionExp):
    model_type = HGRN2Forecaster


class HGRN2ForecasterImputation(ImputationExp):
    model_type = HGRN2Forecaster


class HGRN2ForecasterUEAClassification(UEAClassificationExp):
    model_type = HGRN2Forecaster
