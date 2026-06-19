from torch_timeseries.experiments.forecast import ForecastExp
from torch_timeseries.experiments.anomaly_detection import AnomalyDetectionExp
from torch_timeseries.experiments.imputation import ImputationExp
from torch_timeseries.experiments.uea_classification import UEAClassificationExp
from torch_timeseries.model.QRNNForecaster import QRNNForecaster


class QRNNForecasterForecast(ForecastExp):
    model_type = QRNNForecaster


class QRNNForecasterAnomalyDetection(AnomalyDetectionExp):
    model_type = QRNNForecaster


class QRNNForecasterImputation(ImputationExp):
    model_type = QRNNForecaster


class QRNNForecasterUEAClassification(UEAClassificationExp):
    model_type = QRNNForecaster
