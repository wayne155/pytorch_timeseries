import os
import glob

from .forecast import ForecastExp
from .uea_classification import UEAClassificationExp
from .anomaly_detection import AnomalyDetectionExp
from .imputation import ImputationExp
import importlib

from .DLinear import (
    DLinearAnomalyDetection,
    DLinearForecast,
    DLinearImputation,
    DLinearUEAClassification,
)

from .Autoformer import (
    AutoformerAnomalyDetection,
    AutoformerForecast,
    AutoformerImputation,
    AutoformerUEAClassification,
)

from .FEDformer import (
    FEDformerAnomalyDetection,
    FEDformerForecast,
    FEDformerImputation,
    FEDformerUEAClassification,
)

from .Informer import (
    InformerAnomalyDetection,
    InformerForecast,
    InformerImputation,
    InformerUEAClassification,
)

from .PatchTST import (
    PatchTSTAnomalyDetection,
    PatchTSTForecast,
    PatchTSTImputation,
    PatchTSTUEAClassification,
)


from .CATS import CATSForecast

model_list = ['iTransformer','TSMixer','TimesNet', 'SCINet', 'Crossformer']

class_suffixes = ['AnomalyDetection', 'Forecast', 'Imputation', 'UEAClassification']

for model_prefix in model_list:
    for suffix in class_suffixes:
        module_name = f"{model_prefix}{suffix}"
        try:
            module = importlib.import_module(f".{model_prefix}", package=__name__)
            
            model_class = getattr(module, module_name, None)
            
            if model_class:
                globals()[module_name] = model_class
        except ModuleNotFoundError:
            print(f"Module {module_name} not found.")
