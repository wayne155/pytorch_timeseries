"""Tests for the TCNForecaster model."""
import pytest
import torch

from torch_timeseries.model.TCNForecaster import TCNForecaster

B, L, C, H = 4, 96, 7, 24


class TestTCNForecasterShape:
    def test_forecast_output_shape(self):
        model = TCNForecaster(seq_len=L, pred_len=H, enc_in=C)
        out = model(torch.randn(B, L, C))
        assert out.shape == (B, H, C)

    def test_imputation_output_shape(self):
        model = TCNForecaster(seq_len=L, pred_len=L, enc_in=C)
        out = model(torch.randn(B, L, C))
        assert out.shape == (B, L, C)

    def test_classification_output_shape(self):
        num_classes = 10
        model = TCNForecaster(seq_len=L, pred_len=L, enc_in=C,
                              output_prob=num_classes, revin=False)
        out = model(torch.randn(B, L, C))
        assert out.shape == (B, num_classes)

    def test_single_channel(self):
        model = TCNForecaster(seq_len=L, pred_len=H, enc_in=1)
        out = model(torch.randn(B, L, 1))
        assert out.shape == (B, H, 1)

    def test_single_timestep_pred(self):
        model = TCNForecaster(seq_len=L, pred_len=1, enc_in=C)
        out = model(torch.randn(B, L, C))
        assert out.shape == (B, 1, C)


class TestTCNForecasterGradients:
    def test_gradients_flow_through_output(self):
        model = TCNForecaster(seq_len=L, pred_len=H, enc_in=C)
        x = torch.randn(B, L, C, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None

    def test_model_parameters_receive_gradients(self):
        model = TCNForecaster(seq_len=L, pred_len=H, enc_in=C)
        x = torch.randn(B, L, C)
        out = model(x)
        out.sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"


class TestTCNForecasterProperties:
    def test_no_nan_output(self):
        model = TCNForecaster(seq_len=L, pred_len=H, enc_in=C)
        out = model(torch.randn(B, L, C))
        assert not torch.isnan(out).any()

    def test_revin_disabled(self):
        model = TCNForecaster(seq_len=L, pred_len=H, enc_in=C, revin=False)
        assert not hasattr(model, "rev")
        out = model(torch.randn(B, L, C))
        assert out.shape == (B, H, C)

    def test_revin_enabled_by_default(self):
        model = TCNForecaster(seq_len=L, pred_len=H, enc_in=C)
        assert hasattr(model, "rev")

    def test_different_d_model(self):
        for d in [16, 32, 128]:
            model = TCNForecaster(seq_len=L, pred_len=H, enc_in=C, d_model=d)
            out = model(torch.randn(B, L, C))
            assert out.shape == (B, H, C)

    def test_different_num_levels(self):
        for n in [1, 2, 6]:
            model = TCNForecaster(seq_len=L, pred_len=H, enc_in=C, num_levels=n)
            out = model(torch.randn(B, L, C))
            assert out.shape == (B, H, C)

    def test_different_kernel_size(self):
        for k in [2, 3, 5]:
            model = TCNForecaster(seq_len=L, pred_len=H, enc_in=C, kernel_size=k)
            out = model(torch.randn(B, L, C))
            assert out.shape == (B, H, C)

    def test_zero_dropout(self):
        model = TCNForecaster(seq_len=L, pred_len=H, enc_in=C, dropout=0.0)
        model.eval()
        x = torch.randn(B, L, C)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_eval_mode_deterministic(self):
        model = TCNForecaster(seq_len=L, pred_len=H, enc_in=C, dropout=0.5)
        model.eval()
        x = torch.randn(B, L, C)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)
