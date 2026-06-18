"""Tests for torch_timeseries.utils — EarlyStopping and model_summary."""
import math
import os
import tempfile

import pytest
import torch
import torch.nn as nn

from torch_timeseries.utils import EarlyStopping, count_parameters, model_summary


# ── helpers ───────────────────────────────────────────────────────────────────

def _small_model():
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


# ── EarlyStopping ─────────────────────────────────────────────────────────────

class TestEarlyStopping:
    def test_no_stop_while_improving(self, tmp_path):
        es = EarlyStopping(patience=3, verbose=False,
                           path=str(tmp_path / "ckpt.pt"), trace_func=lambda x: None)
        model = _small_model()
        for val_loss in [1.0, 0.9, 0.8, 0.7]:
            es(val_loss, model)
        assert not es.early_stop

    def test_stops_after_patience(self, tmp_path):
        es = EarlyStopping(patience=3, verbose=False,
                           path=str(tmp_path / "ckpt.pt"), trace_func=lambda x: None)
        model = _small_model()
        es(1.0, model)           # sets best_score = -1.0
        for _ in range(3):
            es(1.5, model)       # strictly worse → counter increments
        assert es.early_stop

    def test_counter_increments_on_no_improvement(self, tmp_path):
        es = EarlyStopping(patience=5, verbose=False,
                           path=str(tmp_path / "ckpt.pt"), trace_func=lambda x: None)
        model = _small_model()
        es(1.0, model)           # best score set
        es(1.1, model)           # worse → +1
        es(1.2, model)           # worse → +2
        assert es.counter == 2

    def test_counter_resets_on_improvement(self, tmp_path):
        es = EarlyStopping(patience=5, verbose=False,
                           path=str(tmp_path / "ckpt.pt"), trace_func=lambda x: None)
        model = _small_model()
        es(1.0, model)
        es(1.0, model)   # counter = 1
        es(0.5, model)   # improvement → counter = 0
        assert es.counter == 0

    def test_checkpoint_saved_on_improvement(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        es = EarlyStopping(patience=3, verbose=False,
                           path=path, trace_func=lambda x: None)
        model = _small_model()
        es(1.0, model)
        assert os.path.exists(path)

    def test_checkpoint_loadable(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        es = EarlyStopping(patience=3, verbose=False,
                           path=path, trace_func=lambda x: None)
        model = _small_model()
        es(1.0, model)
        state = torch.load(path, weights_only=True)
        model2 = _small_model()
        model2.load_state_dict(state)

    def test_reset_clears_state(self, tmp_path):
        es = EarlyStopping(patience=3, verbose=False,
                           path=str(tmp_path / "ckpt.pt"), trace_func=lambda x: None)
        model = _small_model()
        es(1.0, model)
        es(1.0, model)
        es(1.0, model)
        es.reset()
        assert es.counter == 0
        assert es.best_score is None
        assert not es.early_stop

    def test_get_set_state_roundtrip(self, tmp_path):
        es = EarlyStopping(patience=5, verbose=False,
                           path=str(tmp_path / "ckpt.pt"), trace_func=lambda x: None)
        model = _small_model()
        es(1.0, model)
        es(1.1, model)   # strictly worse → counter = 1
        state = es.get_state()
        assert state["counter"] == 1
        es.set_state(state)
        assert es.counter == 1

    def test_delta_requires_strict_improvement(self, tmp_path):
        es = EarlyStopping(patience=3, delta=0.1, verbose=False,
                           path=str(tmp_path / "ckpt.pt"), trace_func=lambda x: None)
        model = _small_model()
        es(1.0, model)
        # 0.95 is better than 1.0 but improvement (0.05) < delta (0.1) → not an improvement
        es(0.95, model)
        assert es.counter == 1

    def test_patience_1_stops_after_one_stagnation(self, tmp_path):
        es = EarlyStopping(patience=1, verbose=False,
                           path=str(tmp_path / "ckpt.pt"), trace_func=lambda x: None)
        model = _small_model()
        es(1.0, model)
        es(1.1, model)   # strictly worse → counter=1 >= patience=1
        assert es.early_stop


# ── model_summary ─────────────────────────────────────────────────────────────

class TestModelSummary:
    def test_returns_correct_keys(self):
        info = model_summary(_small_model())
        assert "total_params" in info
        assert "trainable_params" in info
        assert "non_trainable_params" in info
        assert "size_mb" in info
        assert "param_table" in info

    def test_total_params_correct(self):
        model = nn.Linear(8, 4)   # 8*4 + 4 = 36 params
        info = model_summary(model)
        assert info["total_params"] == 36
        assert info["trainable_params"] == 36
        assert info["non_trainable_params"] == 0

    def test_frozen_params_counted(self):
        model = nn.Linear(8, 4)
        for p in model.parameters():
            p.requires_grad_(False)
        info = model_summary(model)
        assert info["non_trainable_params"] == 36
        assert info["trainable_params"] == 0

    def test_mixed_frozen_trainable(self):
        model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
        for p in model[0].parameters():
            p.requires_grad_(False)
        info = model_summary(model)
        assert info["non_trainable_params"] > 0
        assert info["trainable_params"] > 0
        assert info["total_params"] == info["non_trainable_params"] + info["trainable_params"]

    def test_size_mb_positive(self):
        info = model_summary(_small_model())
        assert info["size_mb"] > 0

    def test_size_mb_float32_estimate(self):
        model = nn.Linear(256, 256)   # 256*256 + 256 = 65792 params
        info = model_summary(model)
        expected_mb = 65792 * 4 / 1024 / 1024
        assert math.isclose(info["size_mb"], expected_mb, rel_tol=1e-5)


# ── count_parameters ──────────────────────────────────────────────────────────

class TestCountParameters:
    def test_returns_table_and_count(self):
        table, count = count_parameters(_small_model())
        assert count > 0

    def test_frozen_params_excluded(self):
        model = nn.Linear(8, 4)
        for p in model.parameters():
            p.requires_grad_(False)
        _, count = count_parameters(model)
        assert count == 0

    def test_count_matches_manual(self):
        model = nn.Linear(8, 4)
        _, count = count_parameters(model)
        expected = 8 * 4 + 4
        assert count == expected
