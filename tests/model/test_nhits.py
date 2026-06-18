import torch


def test_nhits_forecast_shape():
    from torch_timeseries.model.NHiTS import NHiTS
    m = NHiTS(seq_len=96, pred_len=96, enc_in=7)
    out = m(torch.randn(4, 96, 7))
    assert out.shape == (4, 96, 7)


def test_nhits_long_horizon():
    from torch_timeseries.model.NHiTS import NHiTS
    m = NHiTS(seq_len=96, pred_len=720, enc_in=7)
    out = m(torch.randn(4, 96, 7))
    assert out.shape == (4, 720, 7)


def test_nhits_classification():
    from torch_timeseries.model.NHiTS import NHiTS
    m = NHiTS(seq_len=96, pred_len=96, enc_in=7, output_prob=4)
    out = m(torch.randn(4, 96, 7))
    assert out.shape == (4, 4)


def test_nhits_forecast_gradients_flow():
    """Forecast theta parameters must receive gradients."""
    from torch_timeseries.model.NHiTS import NHiTS
    m = NHiTS(seq_len=96, pred_len=96, enc_in=3, n_stacks=2, mlp_units=32, n_theta=16, dropout=0.0)
    m(torch.randn(2, 96, 3)).sum().backward()
    # theta_f and basis_f always receive gradients; theta_b of the last block may not
    for name, p in m.named_parameters():
        if p.requires_grad and "theta_f" in name or "basis_f" in name:
            assert p.grad is not None, f"{name} has no gradient"


def test_nhits_no_nan():
    from torch_timeseries.model.NHiTS import NHiTS
    m = NHiTS(seq_len=96, pred_len=96, enc_in=3, n_stacks=2)
    out = m(torch.randn(2, 96, 3))
    assert not torch.isnan(out).any()


def test_nhits_registry():
    from torch_timeseries.experiments import get_experiment_class
    for task in ["Forecast", "UEAClassification", "AnomalyDetection", "Imputation"]:
        cls = get_experiment_class("NHiTS", task)
        assert cls is not None
