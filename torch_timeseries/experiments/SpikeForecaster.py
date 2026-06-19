from torch_timeseries.experiments.forecast import ForecastExp
from torch_timeseries.experiments.anomaly_detection import AnomalyDetectionExp
from torch_timeseries.experiments.imputation import ImputationExp
from torch_timeseries.experiments.uea_classification import UEAClassificationExp
from torch_timeseries.model.SpikeForecaster import SpikeForecaster


class SpikeForecasterForecast(ForecastExp):
    model_type = SpikeForecaster


class SpikeForecasterAnomalyDetection(AnomalyDetectionExp):
    model_type = SpikeForecaster


class SpikeForecasterImputation(ImputationExp):
    model_type = SpikeForecaster


class SpikeForecasterUEAClassification(UEAClassificationExp):
    model_type = SpikeForecaster
