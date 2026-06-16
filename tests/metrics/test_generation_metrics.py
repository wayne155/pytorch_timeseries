import torch
import numpy as np
import pytest
from torch_timeseries.metrics.generation import (
    discriminative_score,
    predictive_score,
    context_fid,
    correlational_score,
)


def _rand(N=50, T=16, C=3):
    return torch.randn(N, T, C)


def test_discriminative_score_identical():
    x = _rand()
    score = discriminative_score(x, x.clone(), n_runs=1, epochs=2)
    assert isinstance(score, float)
    assert 0.0 <= score <= 0.5


def test_discriminative_score_separable():
    real = torch.zeros(50, 16, 3)
    fake = torch.ones(50, 16, 3)
    score = discriminative_score(real, fake, n_runs=1, epochs=10)
    assert score > 0.1


def test_predictive_score_finite():
    real, fake = _rand(), _rand()
    score = predictive_score(real, fake, epochs=2)
    assert np.isfinite(score)
    assert score >= 0


def test_context_fid_identical():
    x = _rand()
    score = context_fid(x, x.clone())
    assert isinstance(score, float)
    assert score >= 0


def test_context_fid_different():
    real = _rand()
    fake = _rand() * 10
    score = context_fid(real, fake)
    assert score > 0


def test_correlational_score_identical():
    x = _rand()
    score = correlational_score(x, x.clone())
    assert score < 1e-6


def test_correlational_score_positive():
    real, fake = _rand(), _rand()
    score = correlational_score(real, fake)
    assert score >= 0
