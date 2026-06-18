"""Tests for the RNNForecaster model (GRU/LSTM/RNN)."""
import pytest
import torch

from torch_timeseries.model.RNNForecaster import RNNForecaster

B, L, C, H = 4, 96, 7, 24


class TestRNNForecasterShape:
    @pytest.mark.parametrize("rnn_type", ["gru", "lstm", "rnn"])
    def test_forecast_output_shape(self, rnn_type):
        model = RNNForecaster(seq_len=L, pred_len=H, enc_in=C, rnn_type=rnn_type)
        out = model(torch.randn(B, L, C))
        assert out.shape == (B, H, C)

    def test_imputation_output_shape(self):
        model = RNNForecaster(seq_len=L, pred_len=L, enc_in=C)
        assert model(torch.randn(B, L, C)).shape == (B, L, C)

    def test_classification_output_shape(self):
        model = RNNForecaster(seq_len=L, pred_len=L, enc_in=C,
                               output_prob=10, revin=False)
        assert model(torch.randn(B, L, C)).shape == (B, 10)

    def test_single_channel(self):
        model = RNNForecaster(seq_len=L, pred_len=H, enc_in=1)
        assert model(torch.randn(B, L, 1)).shape == (B, H, 1)

    def test_bidirectional(self):
        model = RNNForecaster(seq_len=L, pred_len=H, enc_in=C, bidirectional=True)
        assert model(torch.randn(B, L, C)).shape == (B, H, C)

    def test_single_layer(self):
        model = RNNForecaster(seq_len=L, pred_len=H, enc_in=C, num_layers=1)
        assert model(torch.randn(B, L, C)).shape == (B, H, C)

    def test_deep_model(self):
        model = RNNForecaster(seq_len=L, pred_len=H, enc_in=C, num_layers=4)
        assert model(torch.randn(B, L, C)).shape == (B, H, C)


class TestRNNForecasterGradients:
    def test_gradients_flow(self):
        model = RNNForecaster(seq_len=L, pred_len=H, enc_in=C)
        x = torch.randn(B, L, C, requires_grad=True)
        model(x).sum().backward()
        assert x.grad is not None

    def test_model_params_receive_gradients(self):
        model = RNNForecaster(seq_len=L, pred_len=H, enc_in=C)
        x = torch.randn(B, L, C)
        model(x).sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"


class TestRNNForecasterProperties:
    def test_no_nan(self):
        model = RNNForecaster(seq_len=L, pred_len=H, enc_in=C)
        assert not torch.isnan(model(torch.randn(B, L, C))).any()

    def test_revin_disabled(self):
        model = RNNForecaster(seq_len=L, pred_len=H, enc_in=C, revin=False)
        assert not hasattr(model, "rev")
        assert model(torch.randn(B, L, C)).shape == (B, H, C)

    def test_eval_mode_deterministic(self):
        model = RNNForecaster(seq_len=L, pred_len=H, enc_in=C, dropout=0.5)
        model.eval()
        x = torch.randn(B, L, C)
        assert torch.allclose(model(x), model(x))

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm", "rnn"])
    def test_no_nan_for_all_types(self, rnn_type):
        model = RNNForecaster(seq_len=L, pred_len=H, enc_in=C, rnn_type=rnn_type)
        assert not torch.isnan(model(torch.randn(B, L, C))).any()

    def test_larger_hidden_size(self):
        model = RNNForecaster(seq_len=L, pred_len=H, enc_in=C, hidden_size=256)
        assert model(torch.randn(B, L, C)).shape == (B, H, C)
