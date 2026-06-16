# tests/experiments/test_generation_experiments.py
import numpy as np
import pandas as pd
import pytest


class _ToyDS:
    name = "__toy_gen__"
    freq = "h"
    def __init__(self, T=120, C=2):
        self.data = np.random.randn(T, C).astype(np.float32)
        self.num_features = C
        self.length = T
        self.dates = pd.DataFrame(
            {"date": pd.date_range("2020-01-01", periods=T, freq="h")}
        )


def _run(cls, **kwargs):
    ds = _ToyDS()
    exp = cls(
        dataset_type="__toy_gen__",
        seq_len=8, epochs=2, patience=100,
        batch_size=8, eval_n_samples=16,
        device="cpu", **kwargs,
    )
    exp._toy_dataset = ds
    result = exp.run(seed=0)
    for k in ("discriminative_score", "predictive_score",
              "context_fid", "correlational_score"):
        assert k in result and np.isfinite(result[k])
    return result


def test_timegan_smoke():
    from torch_timeseries.experiments.TimeGAN import TimeGANGeneration
    _run(TimeGANGeneration, hidden_dim=4, n_layers=2,
         epochs_ae=2, epochs_sup=2, epochs_joint=2)


def test_csdi_smoke():
    from torch_timeseries.experiments.CSDI import CSDIGeneration
    _run(CSDIGeneration, d_model=8, n_heads=2, n_layers=1, T=3)


def test_diffusion_ts_smoke():
    from torch_timeseries.experiments.DiffusionTS import DiffusionTSGeneration
    _run(DiffusionTSGeneration, d_model=8, n_heads=2, n_layers=1, T=3)


def test_timediff_smoke():
    from torch_timeseries.experiments.TimeDiff import TimeDiffGeneration
    _run(TimeDiffGeneration, d_model=8, n_heads=2, n_layers=1, T=3)


def test_ns_diffusion_smoke():
    from torch_timeseries.experiments.NSDiffusion import NSDiffusionGeneration
    _run(NSDiffusionGeneration, d_model=8, n_heads=2, n_layers=1, T=3)
