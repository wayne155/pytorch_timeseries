# tests/model/test_generation_models.py
import torch
import pytest


def test_timegan_forward():
    from torch_timeseries.model.TimeGAN import TimeGAN
    m = TimeGAN(seq_len=8, n_features=3, hidden_dim=4, n_layers=2)
    x = torch.randn(2, 8, 3)
    h = m.embed(x)
    assert h.shape == (2, 8, 4)
    x_hat = m.recover(h)
    assert x_hat.shape == (2, 8, 3)


def test_timegan_generate():
    from torch_timeseries.model.TimeGAN import TimeGAN
    m = TimeGAN(seq_len=8, n_features=3, hidden_dim=4, n_layers=2)
    out = m.generate(5)
    assert out.shape == (5, 8, 3)
