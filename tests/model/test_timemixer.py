import pytest
import torch


def test_timemixer_forecast_shape():
    from torch_timeseries.model.TimeMixer import TimeMixer
    m = TimeMixer(seq_len=96, pred_len=96, enc_in=7)
    x = torch.randn(4, 96, 7)
    out = m(x)
    assert out.shape == (4, 96, 7)


def test_timemixer_long_horizon():
    from torch_timeseries.model.TimeMixer import TimeMixer
    m = TimeMixer(seq_len=96, pred_len=720, enc_in=7)
    x = torch.randn(4, 96, 7)
    out = m(x)
    assert out.shape == (4, 720, 7)


def test_timemixer_classification():
    from torch_timeseries.model.TimeMixer import TimeMixer
    m = TimeMixer(seq_len=96, pred_len=96, enc_in=7, output_prob=4)
    x = torch.randn(4, 96, 7)
    out = m(x)
    assert out.shape == (4, 4)


def test_timemixer_no_nan():
    from torch_timeseries.model.TimeMixer import TimeMixer
    m = TimeMixer(seq_len=96, pred_len=96, enc_in=3, e_layers=2)
    x = torch.randn(2, 96, 3)
    out = m(x)
    assert not torch.isnan(out).any()


def test_timemixer_gradients_flow():
    from torch_timeseries.model.TimeMixer import TimeMixer
    m = TimeMixer(seq_len=96, pred_len=96, enc_in=4, d_model=16, e_layers=2, dropout=0.0)
    x = torch.randn(2, 96, 4)
    m(x).sum().backward()
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} has no gradient"


def test_timemixer_registry():
    from torch_timeseries.experiments import get_experiment_class
    for task in ["Forecast", "UEAClassification", "AnomalyDetection", "Imputation"]:
        cls = get_experiment_class("TimeMixer", task)
        assert cls is not None
