"""Tests for the PatchMixer model."""
import pytest
import torch

from torch_timeseries.model.PatchMixer import PatchMixer

B, L, C, H = 4, 96, 7, 24


class TestPatchMixerShape:
    def test_forecast_output_shape(self):
        model = PatchMixer(seq_len=L, pred_len=H, enc_in=C)
        out = model(torch.randn(B, L, C))
        assert out.shape == (B, H, C)

    def test_imputation_output_shape(self):
        model = PatchMixer(seq_len=L, pred_len=L, enc_in=C)
        out = model(torch.randn(B, L, C))
        assert out.shape == (B, L, C)

    def test_classification_output_shape(self):
        model = PatchMixer(seq_len=L, pred_len=L, enc_in=C,
                            output_prob=10, revin=False)
        out = model(torch.randn(B, L, C))
        assert out.shape == (B, 10)

    def test_single_channel(self):
        model = PatchMixer(seq_len=L, pred_len=H, enc_in=1, patch_len=16, stride=8)
        out = model(torch.randn(B, L, 1))
        assert out.shape == (B, H, 1)

    def test_short_pred_len(self):
        model = PatchMixer(seq_len=L, pred_len=1, enc_in=C)
        out = model(torch.randn(B, L, C))
        assert out.shape == (B, 1, C)

    def test_different_patch_configs(self):
        for patch_len, stride in [(8, 4), (16, 8), (32, 16)]:
            model = PatchMixer(seq_len=L, pred_len=H, enc_in=C,
                                patch_len=patch_len, stride=stride)
            out = model(torch.randn(B, L, C))
            assert out.shape == (B, H, C)


class TestPatchMixerGradients:
    def test_gradients_flow(self):
        model = PatchMixer(seq_len=L, pred_len=H, enc_in=C, depth=2)
        x = torch.randn(B, L, C, requires_grad=True)
        model(x).sum().backward()
        assert x.grad is not None

    def test_model_params_receive_gradients(self):
        model = PatchMixer(seq_len=L, pred_len=H, enc_in=C, depth=2)
        x = torch.randn(B, L, C)
        model(x).sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"


class TestPatchMixerProperties:
    def test_no_nan(self):
        model = PatchMixer(seq_len=L, pred_len=H, enc_in=C)
        out = model(torch.randn(B, L, C))
        assert not torch.isnan(out).any()

    def test_revin_disabled(self):
        model = PatchMixer(seq_len=L, pred_len=H, enc_in=C, revin=False)
        assert not hasattr(model, "rev")
        out = model(torch.randn(B, L, C))
        assert out.shape == (B, H, C)

    def test_depth_1(self):
        model = PatchMixer(seq_len=L, pred_len=H, enc_in=C, depth=1)
        out = model(torch.randn(B, L, C))
        assert out.shape == (B, H, C)

    def test_depth_6(self):
        model = PatchMixer(seq_len=L, pred_len=H, enc_in=C, depth=6)
        out = model(torch.randn(B, L, C))
        assert out.shape == (B, H, C)

    def test_eval_mode_deterministic(self):
        model = PatchMixer(seq_len=L, pred_len=H, enc_in=C, dropout=0.5)
        model.eval()
        x = torch.randn(B, L, C)
        assert torch.allclose(model(x), model(x))

    def test_no_nan_with_constant_input(self):
        model = PatchMixer(seq_len=L, pred_len=H, enc_in=C)
        model.eval()
        x = torch.ones(B, L, C)
        out = model(x)
        assert not torch.isnan(out).any()
