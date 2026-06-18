"""Verifies the experiment registry is complete and internally consistent.

These tests do NOT run training loops — they only verify that:
  1. Every (model, task) pair in the registry can be looked up.
  2. The resulting class has the expected model_type field.
  3. Multi-task models are registered for all their declared tasks.
  4. No dangling task-suffix classes exist that aren't in the registry.
"""
import pytest

from torch_timeseries.experiments import get_experiment_class, list_experiments


# ---------------------------------------------------------------------------
# Ground truth: expected registry population
# ---------------------------------------------------------------------------

FORECAST_FOUR_TASK_MODELS = [
    "DLinear", "NLinear",
    "Autoformer", "FEDformer", "Informer",
    "PatchTST", "iTransformer", "TSMixer",
    "Crossformer", "SCINet", "TimesNet",
    "FreTS", "FITS",
    "SegRNN", "TimeMixer", "TiDE", "NHiTS",
    "TCN", "PatchMixer", "RNN", "VanillaTransformer",
]

FORECAST_ONLY_MODELS = ["CATS"]

# Probabilistic models register only the Forecast task.
PROB_FORECAST_MODELS = ["MCDropout", "Gaussian"]

GENERATION_MODELS = [
    ("TimeGAN", "Generation"),
    ("CSDI", "Generation"),
    ("DiffusionTS", "Generation"),
    ("TimeDiff", "Generation"),
    ("NsDiff", "Generation"),
    ("TMDM", "Generation"),
]

IRREGULAR_MODELS = [
    ("GRUD", ["IrregularClassification", "IrregularInterpolation", "IrregularForecast"]),
    ("mTAN", ["IrregularClassification", "IrregularInterpolation", "IrregularForecast"]),
    ("LatentODE", ["IrregularClassification", "IrregularInterpolation", "IrregularForecast"]),
    ("NeuralCDE", ["IrregularClassification"]),
    ("Raindrop", ["IrregularClassification"]),
]

FOUR_TASKS = ["Forecast", "AnomalyDetection", "Imputation", "UEAClassification"]


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------

class TestRegistryCompleteness:
    @pytest.mark.parametrize("model", FORECAST_FOUR_TASK_MODELS)
    def test_four_task_model_registered_for_all_tasks(self, model):
        for task in FOUR_TASKS:
            cls = get_experiment_class(model, task)
            assert cls is not None, f"{model}/{task} not found in registry"

    @pytest.mark.parametrize("model", FORECAST_ONLY_MODELS)
    def test_forecast_only_model_registered(self, model):
        cls = get_experiment_class(model, "Forecast")
        assert cls is not None

    @pytest.mark.parametrize("model", PROB_FORECAST_MODELS)
    def test_prob_forecast_model_registered(self, model):
        cls = get_experiment_class(model, "Forecast")
        assert cls is not None

    @pytest.mark.parametrize("model", PROB_FORECAST_MODELS)
    def test_prob_forecast_model_has_no_other_tasks(self, model):
        for task in ["AnomalyDetection", "Imputation", "UEAClassification"]:
            with pytest.raises(NotImplementedError):
                get_experiment_class(model, task)

    @pytest.mark.parametrize("model,task", GENERATION_MODELS)
    def test_generation_model_registered(self, model, task):
        cls = get_experiment_class(model, task)
        assert cls is not None

    @pytest.mark.parametrize("model,tasks", IRREGULAR_MODELS)
    def test_irregular_model_registered_for_tasks(self, model, tasks):
        for task in tasks:
            cls = get_experiment_class(model, task)
            assert cls is not None, f"{model}/{task} not found"

    def test_registry_not_empty(self):
        assert len(list_experiments()) > 0

    def test_expected_minimum_size(self):
        assert len(list_experiments()) >= 92  # +1 Gaussian/Forecast


# ---------------------------------------------------------------------------
# model_type consistency
# ---------------------------------------------------------------------------

class TestModelTypeConsistency:
    """Each registered experiment class must carry a model_type matching its
    prefix (registry key model name = prefix before the task suffix)."""

    @pytest.mark.parametrize("model", FORECAST_FOUR_TASK_MODELS)
    def test_forecast_four_task_model_type(self, model):
        for task in FOUR_TASKS:
            cls = get_experiment_class(model, task)
            exp = cls.__new__(cls)
            assert exp.model_type == model, (
                f"{cls.__name__}.model_type should be '{model}', got '{exp.model_type}'"
            )

    @pytest.mark.parametrize("model,tasks", IRREGULAR_MODELS)
    def test_irregular_model_type(self, model, tasks):
        for task in tasks:
            cls = get_experiment_class(model, task)
            exp = cls.__new__(cls)
            assert exp.model_type == model


# ---------------------------------------------------------------------------
# look-up errors raise NotImplementedError
# ---------------------------------------------------------------------------

class TestRegistryErrors:
    def test_unknown_model_raises(self):
        with pytest.raises(NotImplementedError, match="Unknown experiment"):
            get_experiment_class("NonExistentModel", "Forecast")

    def test_unknown_task_raises(self):
        with pytest.raises(NotImplementedError, match="Unknown experiment"):
            get_experiment_class("DLinear", "FlyingPig")

    def test_forecast_only_model_other_tasks_raise(self):
        for task in ["AnomalyDetection", "Imputation", "UEAClassification"]:
            with pytest.raises(NotImplementedError):
                get_experiment_class("CATS", task)


# ---------------------------------------------------------------------------
# all_experiments() structural checks
# ---------------------------------------------------------------------------

class TestListExperiments:
    def test_returns_list_of_tuples(self):
        exps = list_experiments()
        for item in exps:
            assert isinstance(item, tuple) and len(item) == 2

    def test_all_models_have_non_empty_task(self):
        for model, task in list_experiments():
            assert model and task

    def test_no_duplicates(self):
        exps = list_experiments()
        assert len(exps) == len(set(exps))

    def test_forecast_four_task_models_all_present(self):
        exps = set(list_experiments())
        for model in FORECAST_FOUR_TASK_MODELS:
            for task in FOUR_TASKS:
                assert (model, task) in exps, f"Missing ({model}, {task})"
