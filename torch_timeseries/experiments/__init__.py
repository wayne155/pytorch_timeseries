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

from .FITS import (
    FITSForecast,
    FITSUEAClassification,
)

from .CATS import CATSForecast

model_list = ['iTransformer','TSMixer','TimesNet', 'SCINet', 'Crossformer', 'FITS', 'FreTS']

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

from .irregular_classification import IrregularClassificationExp
from .GRUD import GRUDIrregularClassification

from .registry import build_experiment_registry, format_experiment_choices

EXPERIMENT_REGISTRY = build_experiment_registry(globals())


def get_experiment_class(model_name: str, task_name: str):
    try:
        return EXPERIMENT_REGISTRY[(model_name, task_name)]
    except KeyError as exc:
        choices = format_experiment_choices(EXPERIMENT_REGISTRY)
        raise NotImplementedError(
            f"Unknown experiment: {model_name}{task_name}. "
            f"Available experiments: {choices}"
        ) from exc


def list_experiments():
    return sorted(EXPERIMENT_REGISTRY)
