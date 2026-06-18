import pytest
import numpy as np


class _ToyIrregular:
    num_features = 3
    num_classes = 0

    def __init__(self, n=40):
        rng = np.random.default_rng(0)
        self.samples, self.times, self.masks = [], [], []
        self.labels = None
        for i in range(n):
            T = rng.integers(8, 20)
            self.samples.append(rng.normal(size=(T, 3)).astype("float32"))
            self.times.append(np.sort(rng.uniform(0, 48, T)).astype("float32"))
            self.masks.append((rng.random((T, 3)) > 0.2).astype("float32"))

    def __len__(self):
        return len(self.samples)


def test_grud_irregular_interpolation_single_run(tmp_path):
    from torch_timeseries.experiments.GRUD import GRUDIrregularInterpolation
    exp = GRUDIrregularInterpolation(
        dataset_type="__toy__",
        epochs=2, patience=5, batch_size=8,
        hidden_size=16, device="cpu",
        save_dir=str(tmp_path),
    )
    exp._toy_dataset = _ToyIrregular()
    result = exp.run(seed=1)
    assert isinstance(result, dict)
    assert "mse" in result
    assert result["mse"] >= 0.0


def test_grud_irregular_forecast_single_run(tmp_path):
    from torch_timeseries.experiments.GRUD import GRUDIrregularForecast
    exp = GRUDIrregularForecast(
        dataset_type="__toy__",
        epochs=2, patience=5, batch_size=8,
        hidden_size=16, device="cpu",
        save_dir=str(tmp_path),
    )
    exp._toy_dataset = _ToyIrregular()
    result = exp.run(seed=1)
    assert isinstance(result, dict)
    assert "mse" in result
    assert result["mse"] >= 0.0
