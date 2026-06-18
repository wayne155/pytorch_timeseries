"""Tests for TCNConfig, PatchMixerConfig, RNNConfig, VanillaTransformerConfig."""
import pytest

from torch_timeseries.experiments.configs import (
    PatchMixerConfig,
    RNNConfig,
    RuntimeConfig,
    TCNConfig,
    VanillaTransformerConfig,
    split_experiment_config,
)


class TestTCNConfig:
    def test_split_accepts_flat_kwargs(self):
        w, s, m, r = split_experiment_config(
            model="TCN",
            task="Forecast",
            kwargs={
                "windows": 96, "pred_len": 24,
                "d_model": 32, "num_levels": 3, "kernel_size": 5,
                "dropout": 0.2, "revin": False,
                "epochs": 1, "save_dir": "./tmp",
            },
        )
        assert w.window == 96
        assert w.steps == 24
        assert isinstance(m, TCNConfig)
        assert m.d_model == 32
        assert m.num_levels == 3
        assert m.kernel_size == 5
        assert m.dropout == 0.2
        assert m.revin is False

    def test_rejects_irrelevant_kwargs(self):
        with pytest.raises(TypeError, match="individual"):
            split_experiment_config(
                model="TCN",
                task="Forecast",
                kwargs={"windows": 96, "pred_len": 24, "individual": True},
            )

    def test_validate_d_model_positive(self):
        cfg = TCNConfig(d_model=0)
        with pytest.raises(ValueError, match="d_model"):
            cfg.validate()

    def test_validate_dropout_range(self):
        cfg = TCNConfig(dropout=1.5)
        with pytest.raises(ValueError, match="dropout"):
            cfg.validate()

    def test_defaults_are_valid(self):
        TCNConfig().validate()


class TestPatchMixerConfig:
    def test_split_accepts_flat_kwargs(self):
        w, s, m, r = split_experiment_config(
            model="PatchMixer",
            task="Forecast",
            kwargs={
                "windows": 96, "pred_len": 48,
                "patch_len": 8, "patch_stride": 4, "d_model": 32,
                "depth": 2, "dropout": 0.05, "revin": True,
                "epochs": 1, "save_dir": "./tmp",
            },
        )
        assert isinstance(m, PatchMixerConfig)
        assert m.patch_len == 8
        assert m.patch_stride == 4
        assert m.depth == 2
        assert m.revin is True

    def test_rejects_irrelevant_kwargs(self):
        with pytest.raises(TypeError, match="individual"):
            split_experiment_config(
                model="PatchMixer",
                task="Forecast",
                kwargs={"windows": 96, "pred_len": 24, "individual": True},
            )

    def test_validate_patch_len_positive(self):
        cfg = PatchMixerConfig(patch_len=0)
        with pytest.raises(ValueError, match="patch_len"):
            cfg.validate()

    def test_validate_patch_stride_positive(self):
        cfg = PatchMixerConfig(patch_stride=0)
        with pytest.raises(ValueError, match="patch_stride"):
            cfg.validate()

    def test_defaults_are_valid(self):
        PatchMixerConfig().validate()


class TestRNNConfig:
    def test_split_accepts_flat_kwargs(self):
        w, s, m, r = split_experiment_config(
            model="RNN",
            task="Forecast",
            kwargs={
                "windows": 96, "pred_len": 24,
                "hidden_size": 128, "num_layers": 3,
                "rnn_type": "lstm", "dropout": 0.3,
                "bidirectional": True, "revin": False,
                "epochs": 1, "save_dir": "./tmp",
            },
        )
        assert isinstance(m, RNNConfig)
        assert m.hidden_size == 128
        assert m.num_layers == 3
        assert m.rnn_type == "lstm"
        assert m.bidirectional is True
        assert m.revin is False

    def test_rejects_irrelevant_kwargs(self):
        with pytest.raises(TypeError, match="individual"):
            split_experiment_config(
                model="RNN",
                task="Forecast",
                kwargs={"windows": 96, "pred_len": 24, "individual": True},
            )

    def test_validate_rnn_type(self):
        cfg = RNNConfig(rnn_type="transformer")
        with pytest.raises(ValueError, match="rnn_type"):
            cfg.validate()

    def test_validate_hidden_size_positive(self):
        cfg = RNNConfig(hidden_size=-1)
        with pytest.raises(ValueError, match="hidden_size"):
            cfg.validate()

    def test_all_rnn_types_valid(self):
        for rnn_type in ("gru", "lstm", "rnn"):
            RNNConfig(rnn_type=rnn_type).validate()

    def test_defaults_are_valid(self):
        RNNConfig().validate()


class TestVanillaTransformerConfig:
    def test_split_accepts_flat_kwargs(self):
        w, s, m, r = split_experiment_config(
            model="VanillaTransformer",
            task="Forecast",
            kwargs={
                "windows": 96, "pred_len": 24,
                "d_model": 128, "n_heads": 4, "e_layers": 2, "d_ff": 256,
                "dropout": 0.1, "activation": "gelu", "revin": True,
                "epochs": 1, "save_dir": "./tmp",
            },
        )
        assert w.window == 96
        assert w.steps == 24
        assert isinstance(m, VanillaTransformerConfig)
        assert m.d_model == 128
        assert m.n_heads == 4
        assert m.e_layers == 2

    def test_rejects_irrelevant_kwargs(self):
        with pytest.raises(TypeError, match="individual"):
            split_experiment_config(
                model="VanillaTransformer",
                task="Forecast",
                kwargs={"windows": 96, "pred_len": 24, "individual": True},
            )

    def test_validate_d_model_positive(self):
        cfg = VanillaTransformerConfig(d_model=0)
        with pytest.raises(ValueError, match="d_model"):
            cfg.validate()

    def test_validate_n_heads_divides_d_model(self):
        cfg = VanillaTransformerConfig(d_model=256, n_heads=7)
        with pytest.raises(ValueError, match="n_heads"):
            cfg.validate()

    def test_validate_activation(self):
        cfg = VanillaTransformerConfig(activation="sigmoid")
        with pytest.raises(ValueError, match="activation"):
            cfg.validate()

    def test_validate_dropout_range(self):
        cfg = VanillaTransformerConfig(dropout=1.0)
        with pytest.raises(ValueError, match="dropout"):
            cfg.validate()

    def test_defaults_are_valid(self):
        VanillaTransformerConfig().validate()

    def test_relu_activation_valid(self):
        VanillaTransformerConfig(activation="relu").validate()
