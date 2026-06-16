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


def test_csdi_forward():
    from torch_timeseries.model.CSDI import CSDI
    m = CSDI(seq_len=8, n_features=3, d_model=16, n_heads=2, n_layers=2, T=5)
    x_t = torch.randn(2, 8, 3)
    t   = torch.randint(0, 5, (2,))
    out = m.denoise(x_t, t)
    assert out.shape == (2, 8, 3)


def test_csdi_generate():
    from torch_timeseries.model.CSDI import CSDI
    m = CSDI(seq_len=8, n_features=3, d_model=16, n_heads=2, n_layers=2, T=5)
    out = m.generate(4)
    assert out.shape == (4, 8, 3)
