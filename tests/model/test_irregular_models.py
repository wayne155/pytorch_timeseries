# tests/model/test_irregular_models.py
import pytest
import torch


def _make_synthetic_batch(B=4, T=10, F=3, device="cpu"):
    """Create a padded irregular batch for model testing."""
    x = torch.randn(B, T, F)
    t = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
    mask = (torch.rand(B, T, F) > 0.4).float()
    return x.to(device), t.to(device), mask.to(device)


def test_grud_forward_classification():
    """GRU-D returns (B, num_classes) for classification."""
    from torch_timeseries.model.irregular.grud import GRUD
    B, T, F, C = 4, 10, 3, 2
    model = GRUD(input_size=F, hidden_size=16, output_size=C)
    x, t, mask = _make_synthetic_batch(B, T, F)
    out = model(x, t, mask)
    assert out.shape == (B, C), f"Expected ({B}, {C}), got {out.shape}"


def test_grud_no_nan():
    """GRU-D output has no NaN even with fully-masked (all-missing) inputs."""
    from torch_timeseries.model.irregular.grud import GRUD
    B, T, F, C = 4, 10, 3, 2
    model = GRUD(input_size=F, hidden_size=16, output_size=C)
    x = torch.randn(B, T, F)
    t = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
    mask = torch.zeros(B, T, F)  # all missing
    out = model(x, t, mask)
    assert not torch.isnan(out).any()


def test_grud_gradients_flow():
    """GRU-D gradients flow to all parameters."""
    from torch_timeseries.model.irregular.grud import GRUD
    model = GRUD(input_size=3, hidden_size=16, output_size=2)
    x, t, mask = _make_synthetic_batch(B=2, T=8, F=3)
    logits = model(x, t, mask)
    logits.sum().backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name}"
