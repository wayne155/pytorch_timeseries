import torch


def test_tide_forecast_shape():
    from torch_timeseries.model.TiDE import TiDE
    m = TiDE(seq_len=96, pred_len=96, enc_in=7)
    out = m(torch.randn(4, 96, 7))
    assert out.shape == (4, 96, 7)


def test_tide_long_horizon():
    from torch_timeseries.model.TiDE import TiDE
    m = TiDE(seq_len=96, pred_len=720, enc_in=7)
    out = m(torch.randn(4, 96, 7))
    assert out.shape == (4, 720, 7)


def test_tide_classification():
    from torch_timeseries.model.TiDE import TiDE
    m = TiDE(seq_len=96, pred_len=96, enc_in=7, output_prob=4)
    out = m(torch.randn(4, 96, 7))
    assert out.shape == (4, 4)


def test_tide_gradients_flow():
    from torch_timeseries.model.TiDE import TiDE
    m = TiDE(seq_len=96, pred_len=96, enc_in=4, hidden_size=32, dropout=0.0)
    m(torch.randn(2, 96, 4)).sum().backward()
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} has no gradient"


def test_tide_registry():
    from torch_timeseries.experiments import get_experiment_class
    for task in ["Forecast", "UEAClassification", "AnomalyDetection", "Imputation"]:
        cls = get_experiment_class("TiDE", task)
        assert cls is not None
