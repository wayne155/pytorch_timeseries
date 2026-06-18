import pytest
import numpy as np


class _ToyIrregular:
    num_features = 3
    num_classes = 2

    def __init__(self, n=30, has_labels=True):
        rng = np.random.default_rng(0)
        self.samples, self.times, self.masks = [], [], []
        lbs = []
        for i in range(n):
            T = rng.integers(6, 15)
            self.samples.append(rng.normal(size=(T, 3)).astype("float32"))
            self.times.append(np.sort(rng.uniform(0, 48, T)).astype("float32"))
            self.masks.append((rng.random((T, 3)) > 0.2).astype("float32"))
            lbs.append(i % 2)
        self.labels = np.array(lbs, dtype=np.int64) if has_labels else None

    def __len__(self):
        return len(self.samples)


def test_mtan_classification_exp_runs(tmp_path):
    from torch_timeseries.experiments.mTAN import mTANIrregularClassification
    exp = mTANIrregularClassification(
        dataset_type="__toy__", epochs=2, patience=5, batch_size=8,
        hidden_size=16, num_ref_points=4, num_heads=1,
        device="cpu", save_dir=str(tmp_path),
    )
    exp._toy_dataset = _ToyIrregular(n=30)
    result = exp.run(seed=1)
    assert "accuracy" in result


def test_mtan_interpolation_exp_runs(tmp_path):
    from torch_timeseries.experiments.mTAN import mTANIrregularInterpolation
    exp = mTANIrregularInterpolation(
        dataset_type="__toy__", epochs=2, patience=5, batch_size=8,
        hidden_size=16, num_ref_points=4, num_heads=1,
        device="cpu", save_dir=str(tmp_path),
    )
    ds = _ToyIrregular(n=30, has_labels=False)
    ds.labels = None
    exp._toy_dataset = ds
    result = exp.run(seed=1)
    assert "mse" in result


def test_mtan_registered():
    from torch_timeseries.experiments import get_experiment_class
    cls = get_experiment_class("mTAN", "IrregularClassification")
    assert cls is not None
    cls2 = get_experiment_class("mTAN", "IrregularInterpolation")
    assert cls2 is not None
    cls3 = get_experiment_class("mTAN", "IrregularForecast")
    assert cls3 is not None


def test_all_phase3_registered():
    from torch_timeseries.experiments import get_experiment_class
    expected = [
        ("LatentODE", "IrregularClassification"),
        ("LatentODE", "IrregularInterpolation"),
        ("LatentODE", "IrregularForecast"),
        ("NeuralCDE", "IrregularClassification"),
        ("Raindrop", "IrregularClassification"),
    ]
    for model, task in expected:
        cls = get_experiment_class(model, task)
        assert cls is not None, f"{model}/{task} not in registry"
