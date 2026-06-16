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


def test_diffusion_ts_forward():
    from torch_timeseries.model.DiffusionTS import DiffusionTS
    m = DiffusionTS(seq_len=8, n_features=3, d_model=16, n_heads=2, n_layers=2, T=5)
    x_t = torch.randn(2, 8, 3)
    t   = torch.randint(0, 5, (2,))
    out = m.denoise(x_t, t)
    assert out.shape == (2, 8, 3)


def test_diffusion_ts_generate():
    from torch_timeseries.model.DiffusionTS import DiffusionTS
    m = DiffusionTS(seq_len=8, n_features=3, d_model=16, n_heads=2, n_layers=2, T=5)
    out = m.generate(4)
    assert out.shape == (4, 8, 3)


def test_timediff_forward():
    from torch_timeseries.model.TimeDiff import TimeDiff
    m = TimeDiff(seq_len=8, n_features=3, d_model=16, n_heads=2, n_layers=2, T=5)
    x_t = torch.randn(2, 8, 3)
    t   = torch.randint(0, 5, (2,))
    out = m.denoise(x_t, t)
    assert out.shape == (2, 8, 3)


def test_timediff_generate():
    from torch_timeseries.model.TimeDiff import TimeDiff
    m = TimeDiff(seq_len=8, n_features=3, d_model=16, n_heads=2, n_layers=2, T=5)
    out = m.generate(4)
    assert out.shape == (4, 8, 3)


def test_ns_diffusion_loss():
    from torch_timeseries.model.NsDiff import NsDiff
    m = NsDiff(seq_len=16, n_features=3, T=5, kernel_size=4)
    x = torch.randn(2, 16, 3)
    loss = m.loss(x)
    assert loss.shape == ()
    assert loss.item() > 0


def test_ns_diffusion_generate():
    from torch_timeseries.model.NsDiff import NsDiff
    m = NsDiff(seq_len=16, n_features=3, T=5, kernel_size=4)
    out = m.generate(4)
    assert out.shape == (4, 16, 3)


def test_tmdm_loss():
    from torch_timeseries.model.TMDM import TMDM
    m = TMDM(seq_len=16, n_features=3, T=5)
    x = torch.randn(2, 16, 3)
    loss = m.loss(x)
    assert loss.shape == ()
    assert loss.item() > 0


def test_tmdm_generate():
    from torch_timeseries.model.TMDM import TMDM
    m = TMDM(seq_len=16, n_features=3, T=5)
    out = m.generate(4)
    assert out.shape == (4, 16, 3)
