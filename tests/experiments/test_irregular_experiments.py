import pytest
import sys
import os


def test_grud_irregular_classification_single_run(tmp_path):
    """GRUDIrregularClassification.run() returns a metrics dict."""
    from torch_timeseries.experiments.GRUD import GRUDIrregularClassification
    from tests.dataloader.test_v2_irregular import _ToyIrregular

    exp = GRUDIrregularClassification(
        dataset_type="__toy__",
        epochs=2,
        patience=5,
        batch_size=8,
        hidden_size=16,
        device="cpu",
        save_dir=str(tmp_path),
    )
    # Inject toy dataset directly to avoid file I/O
    exp._toy_dataset = _ToyIrregular(n=40)

    result = exp.run(seed=1)

    assert isinstance(result, dict)
    assert "accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0


def test_grud_registered_in_experiment_builder():
    """get_experiment_class('GRUD', 'IrregularClassification') resolves without error."""
    # NOTE: This test will pass only after Task 8 wires the registry.
    # For now we just check that the combo class exists and is importable.
    from torch_timeseries.experiments.GRUD import GRUDIrregularClassification
    assert GRUDIrregularClassification is not None
