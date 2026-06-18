import torch
import pytest


def _batch(B=4, T=10, F=3):
    x = torch.randn(B, T, F)
    t = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
    mask = (torch.rand(B, T, F) > 0.4).float()
    return x, t, mask


def test_mtan_classification_forward():
    from torch_timeseries.model.irregular.mtan import mTAN
    B, T, F, C = 4, 10, 3, 2
    model = mTAN(input_size=F, hidden_size=32, output_size=C,
                 num_ref_points=8, num_heads=2)
    x, t, mask = _batch(B, T, F)
    out = model(x, t, mask)
    assert out.shape == (B, C), f"Expected ({B}, {C}), got {out.shape}"


def test_mtan_seq2seq_forward():
    from torch_timeseries.model.irregular.mtan import mTAN
    B, T, F, Tq = 4, 10, 3, 5
    model = mTAN(input_size=F, hidden_size=32, output_size=F,
                 num_ref_points=8, num_heads=2)
    x, t, mask = _batch(B, T, F)
    t_query = torch.linspace(0.5, 1.0, Tq).unsqueeze(0).expand(B, -1)
    out = model(x, t, mask, t_query=t_query)
    assert out.shape == (B, Tq, F), f"Expected ({B},{Tq},{F}), got {out.shape}"


def test_mtan_no_nan():
    from torch_timeseries.model.irregular.mtan import mTAN
    B, T, F = 4, 10, 3
    model = mTAN(input_size=F, hidden_size=32, output_size=2, num_ref_points=8)
    x, t, mask = _batch(B, T, F)
    mask = torch.zeros_like(mask)   # all missing
    out = model(x, t, mask)
    assert not torch.isnan(out).any()


def test_mtan_gradients_flow():
    from torch_timeseries.model.irregular.mtan import mTAN
    model = mTAN(input_size=3, hidden_size=16, output_size=2, num_ref_points=4)
    x, t, mask = _batch(B=2, T=8, F=3)
    logits = model(x, t, mask)
    logits.sum().backward()
    for name, p in model.named_parameters():
        if p.requires_grad and "fc_dec" not in name and "time_embed_dec" not in name and "dec_attn" not in name:
            assert p.grad is not None, f"No grad for {name}"
