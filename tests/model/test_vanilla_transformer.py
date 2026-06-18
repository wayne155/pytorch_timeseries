"""Tests for VanillaTransformer encoder-only forecaster."""
import pytest
import torch

from torch_timeseries.model import VanillaTransformer


B, L, C = 2, 48, 7   # batch, seq_len, channels
PRED = 24


class TestVanillaTransformerConstruction:
    def test_default_construction(self):
        m = VanillaTransformer(seq_len=L, pred_len=PRED, enc_in=C)
        assert isinstance(m, torch.nn.Module)

    def test_revin_enabled_by_default(self):
        m = VanillaTransformer(seq_len=L, pred_len=PRED, enc_in=C)
        assert m.revin is True
        assert hasattr(m, "rev")

    def test_revin_disabled(self):
        m = VanillaTransformer(seq_len=L, pred_len=PRED, enc_in=C, revin=False)
        assert m.revin is False
        assert not hasattr(m, "rev")

    def test_classification_disables_revin(self):
        m = VanillaTransformer(seq_len=L, pred_len=PRED, enc_in=C,
                               revin=True, output_prob=4)
        assert m.revin is False

    def test_encoder_layer_count(self):
        for n in (1, 2, 4):
            m = VanillaTransformer(seq_len=L, pred_len=PRED, enc_in=C, e_layers=n)
            assert len(m.encoder.attn_layers) == n

    def test_relu_activation(self):
        m = VanillaTransformer(seq_len=L, pred_len=PRED, enc_in=C, activation="relu")
        assert isinstance(m, torch.nn.Module)

    def test_gelu_activation(self):
        m = VanillaTransformer(seq_len=L, pred_len=PRED, enc_in=C, activation="gelu")
        assert isinstance(m, torch.nn.Module)


class TestVanillaTransformerForecast:
    def _model(self, **kw):
        return VanillaTransformer(seq_len=L, pred_len=PRED, enc_in=C, **kw)

    def _x(self):
        return torch.randn(B, L, C)

    def test_output_shape(self):
        m = self._model()
        y = m(self._x())
        assert y.shape == (B, PRED, C)

    def test_output_is_finite(self):
        m = self._model()
        y = m(self._x())
        assert torch.isfinite(y).all()

    def test_output_shape_no_revin(self):
        m = self._model(revin=False)
        y = m(self._x())
        assert y.shape == (B, PRED, C)

    def test_batch_size_one(self):
        m = self._model()
        x = torch.randn(1, L, C)
        y = m(x)
        assert y.shape == (1, PRED, C)

    def test_large_batch(self):
        m = self._model()
        x = torch.randn(8, L, C)
        y = m(x)
        assert y.shape == (8, PRED, C)

    def test_different_pred_lens(self):
        for pred in (12, 24, 48, 96):
            m = VanillaTransformer(seq_len=L, pred_len=pred, enc_in=C)
            y = m(self._x())
            assert y.shape == (B, pred, C)

    def test_single_channel(self):
        m = VanillaTransformer(seq_len=L, pred_len=PRED, enc_in=1)
        x = torch.randn(B, L, 1)
        y = m(x)
        assert y.shape == (B, PRED, 1)

    def test_gradient_flows(self):
        m = self._model()
        x = torch.randn(B, L, C, requires_grad=True)
        y = m(x)
        y.sum().backward()
        assert x.grad is not None

    def test_no_grad_forward(self):
        m = self._model()
        with torch.no_grad():
            y = m(self._x())
        assert y.shape == (B, PRED, C)

    def test_train_vs_eval_output_deterministic_eval(self):
        m = self._model(dropout=0.5)
        m.eval()
        x = self._x()
        with torch.no_grad():
            y1 = m(x)
            y2 = m(x)
        assert torch.allclose(y1, y2)

    def test_different_d_models(self):
        for d in (64, 128, 256):
            m = VanillaTransformer(seq_len=L, pred_len=PRED, enc_in=C, d_model=d)
            y = m(self._x())
            assert y.shape == (B, PRED, C)

    def test_multi_head(self):
        m = self._model(n_heads=8, d_model=256)
        y = m(self._x())
        assert y.shape == (B, PRED, C)


class TestVanillaTransformerClassification:
    def _model(self, n_classes=5, **kw):
        return VanillaTransformer(seq_len=L, pred_len=PRED, enc_in=C,
                                  output_prob=n_classes, **kw)

    def _x(self):
        return torch.randn(B, L, C)

    def test_output_shape(self):
        m = self._model(n_classes=5)
        y = m(self._x())
        assert y.shape == (B, 5)

    def test_output_shape_binary(self):
        m = self._model(n_classes=2)
        y = m(self._x())
        assert y.shape == (B, 2)

    def test_output_is_finite(self):
        m = self._model()
        y = m(self._x())
        assert torch.isfinite(y).all()

    def test_gradient_flows(self):
        m = self._model()
        x = torch.randn(B, L, C, requires_grad=True)
        y = m(x)
        y.sum().backward()
        assert x.grad is not None


class TestVanillaTransformerRegistry:
    def test_registered_as_vanilla_transformer(self):
        from torch_timeseries.experiments import get_experiment_class
        cls = get_experiment_class("VanillaTransformer", "Forecast")
        assert cls is not None

    def test_all_four_tasks_registered(self):
        from torch_timeseries.experiments import get_experiment_class
        for task in ("Forecast", "UEAClassification", "AnomalyDetection", "Imputation"):
            cls = get_experiment_class("VanillaTransformer", task)
            assert cls is not None

    def test_in_forecasting_models_list(self):
        from torch_timeseries.model import forecasting_models
        assert "VanillaTransformer" in forecasting_models
