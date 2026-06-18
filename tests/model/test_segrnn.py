import pytest
import torch


def test_segrnn_forecast_shape():
    from torch_timeseries.model.SegRNN import SegRNN
    m = SegRNN(seq_len=96, pred_len=96, enc_in=7, d_model=64, seg_len=48)
    x = torch.randn(4, 96, 7)
    out = m(x)
    assert out.shape == (4, 96, 7)


def test_segrnn_long_horizon():
    from torch_timeseries.model.SegRNN import SegRNN
    m = SegRNN(seq_len=96, pred_len=720, enc_in=7, d_model=64, seg_len=48)
    x = torch.randn(4, 96, 7)
    out = m(x)
    assert out.shape == (4, 720, 7)


def test_segrnn_classification():
    from torch_timeseries.model.SegRNN import SegRNN
    m = SegRNN(seq_len=96, pred_len=96, enc_in=7, d_model=64, seg_len=48, output_prob=5)
    x = torch.randn(4, 96, 7)
    out = m(x)
    assert out.shape == (4, 5)


def test_segrnn_odd_pred_len():
    """SegRNN handles pred_len not divisible by default seg_len."""
    from torch_timeseries.model.SegRNN import SegRNN
    m = SegRNN(seq_len=96, pred_len=192, enc_in=3, d_model=32, seg_len=48)
    x = torch.randn(2, 96, 3)
    out = m(x)
    assert out.shape == (2, 192, 3)


def test_segrnn_gradients_flow():
    from torch_timeseries.model.SegRNN import SegRNN
    m = SegRNN(seq_len=96, pred_len=96, enc_in=4, d_model=32, seg_len=48, dropout=0.0)
    x = torch.randn(2, 96, 4)
    loss = m(x).sum()
    loss.backward()
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} has no gradient"


def test_segrnn_registry():
    from torch_timeseries.experiments import get_experiment_class
    for task in ["Forecast", "UEAClassification", "AnomalyDetection", "Imputation"]:
        cls = get_experiment_class("SegRNN", task)
        assert cls is not None
