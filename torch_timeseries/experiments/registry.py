from typing import Dict, Tuple, Type


TASK_SUFFIXES = ("Forecast", "Imputation", "UEAClassification", "AnomalyDetection",
                 "IrregularClassification", "Generation")


def build_experiment_registry(namespace: dict) -> Dict[Tuple[str, str], Type]:
    registry = {}
    for name, value in namespace.items():
        if not isinstance(value, type):
            continue
        for task in TASK_SUFFIXES:
            if name.endswith(task) and len(name) > len(task):
                registry[(name[: -len(task)], task)] = value
                break
    return registry


def format_experiment_choices(registry: Dict[Tuple[str, str], Type]) -> str:
    return ", ".join(f"{model}/{task}" for model, task in sorted(registry))
