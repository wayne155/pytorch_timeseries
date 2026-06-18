"""Tests that models with optional dependencies raise ImportError with install hints."""
import sys
import pytest


def test_latent_ode_import_error_without_torchdiffeq():
    if "torchdiffeq" in sys.modules:
        pytest.skip("torchdiffeq is installed; this test requires it to be absent")
    from torch_timeseries.model.irregular import latent_ode as _mod
    with pytest.raises(ImportError, match="torchdiffeq"):
        _mod.LatentODE(input_size=3, latent_size=8, hidden_size=16, output_size=2)


def test_neural_cde_import_error_without_torchcde():
    if "torchcde" in sys.modules:
        pytest.skip("torchcde is installed; this test requires it to be absent")
    from torch_timeseries.model.irregular import neural_cde as _mod
    with pytest.raises(ImportError, match="torchcde"):
        _mod.NeuralCDE(input_size=3, hidden_size=16, output_size=2)


def test_raindrop_import_error_without_torch_geometric():
    if "torch_geometric" in sys.modules:
        pytest.skip("torch_geometric is installed; this test requires it to be absent")
    from torch_timeseries.model.irregular import raindrop as _mod
    with pytest.raises(ImportError, match="torch_geometric"):
        _mod.Raindrop(input_size=3, hidden_size=16, output_size=2,
                      num_nodes=3, num_heads=1)
