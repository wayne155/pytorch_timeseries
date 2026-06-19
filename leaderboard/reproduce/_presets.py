"""Shared benchmark grids for per-model reproduce scripts."""

from __future__ import annotations

from copy import deepcopy


DEFAULT_SEEDS = [0, 1, 2, 3, 4]

LONG_TERM_DATASETS = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "Electricity",
    "Weather",
    "Traffic",
    "ExchangeRate",
]

SHORT_TERM_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]

IMPUTATION_DATASETS = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "Electricity",
    "Weather",
]

ANOMALY_DATASETS = ["SMD", "MSL", "SMAP", "SWaT", "PSM"]

UEA_DATASETS = [
    "EthanolConcentration",
    "FaceDetection",
    "Handwriting",
    "Heartbeat",
    "JapaneseVowels",
    "PEMS-SF",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigits",
    "UWaveGestureLibrary",
]

IRREGULAR_DATASETS = ["PhysioNet2012", "PhysioNet2019"]

LONG_TERM_GRID = {"pred_len": [96, 192, 336, 720]}
SHORT_TERM_GRID = {"pred_len": [24, 48, 96]}
IMPUTATION_GRID = {"mask_rate": [0.125, 0.25, 0.375, 0.5]}
NO_GRID = {}


MODEL_PARAMS = {
    "SegRNN": {
        "d_model": 512,
        "seg_len": 48,
        "dropout": 0.5,
    },
    "TimeMixer": {
        "n_heads": 4,
        "d_model": 32,
        "e_layers": 3,
        "dropout": 0.1,
        "down_sampling_window": 2,
        "down_sampling_layers": 3,
    },
    "TiDE": {
        "hidden_size": 256,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "decoder_output_dim": 8,
        "dropout": 0.3,
    },
    "NHiTS": {
        "n_stacks": 3,
        "n_blocks": 1,
        "n_theta": 512,
        "mlp_units": 512,
        "n_layers": 2,
        "dropout": 0.1,
    },
    "GRUD": {
        "hidden_size": 64,
        "dropout": 0.3,
    },
    "mTAN": {
        "hidden_size": 64,
        "num_ref_points": 16,
        "num_heads": 4,
        "dropout": 0.3,
    },
    "LatentODE": {
        "hidden_size": 64,
        "latent_size": 32,
        "dropout": 0.3,
    },
    "Autoformer": {
        "d_ff": 2048,
        "factor": 1,
        "activation": "gelu",
        "e_layers": 2,
        "d_layers": 1,
        "output_attention": True,
        "moving_avg": 25,
        "n_heads": 8,
        "d_model": 512,
        "embed": "timeF",
        "dropout": 0.1,
    },
    "Informer": {
        "factor": 5,
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 2,
        "d_layer": 1,
        "d_ff": 512,
        "dropout": 0.05,
        "attn": "prob",
        "embed": "timeF",
        "distil": True,
        "mix": True,
    },
    "Crossformer": {
        "seg_len": 6,
        "win_size": 2,
        "factor": 10,
        "d_model": 256,
        "d_ff": 512,
        "n_heads": 4,
        "e_layers": 3,
        "dropout": 0.2,
    },
    "TSMixer": {
        "n_mixer": 8,
        "dropout": 0.05,
    },
    "SCINet": {
        "hid_size": 1,
        "num_stacks": 1,
        "num_levels": 3,
        "num_decoder_layer": 1,
        "concat_len": 0,
        "groups": 1,
        "kernel": 5,
        "dropout": 0.5,
        "modified": True,
        "RIN": True,
    },
    "VanillaTransformer": {
        "d_model": 256,
        "n_heads": 4,
        "e_layers": 3,
        "d_ff": 512,
        "dropout": 0.1,
        "activation": "gelu",
        "revin": True,
    },
    "Gaussian": {
        "d_model": 256,
        "n_heads": 4,
        "e_layers": 3,
        "d_ff": 512,
        "dropout": 0.1,
        "activation": "gelu",
        "revin": True,
        "num_samples": 50,
    },
    "MCDropout": {
        "d_model": 256,
        "n_heads": 4,
        "e_layers": 3,
        "d_ff": 512,
        "dropout": 0.1,
        "activation": "gelu",
        "revin": True,
        "num_samples": 50,
    },
    "StudentT": {
        "d_model": 256,
        "n_heads": 4,
        "e_layers": 3,
        "d_ff": 512,
        "dropout": 0.1,
        "activation": "gelu",
        "revin": True,
        "num_samples": 50,
    },
    "Quantile": {
        "d_model": 256,
        "n_heads": 4,
        "e_layers": 3,
        "d_ff": 512,
        "dropout": 0.1,
        "activation": "gelu",
        "revin": True,
    },
    "NBEATS": {
        "stack_types": ["trend", "seasonality", "generic"],
        "num_blocks": 3,
        "hidden_size": 256,
        "expansion_coefficient_dim": 32,
        "degree_of_polynomial": 3,
        "num_harmonics": 4,
    },
    "SparseTSF": {
        "period": 24,   # daily period for hourly data (ETT, Weather, Traffic)
        "revin": True,
    },
    "Ensemble": {
        "d_model": 256,
        "n_heads": 4,
        "e_layers": 3,
        "d_ff": 512,
        "dropout": 0.1,
        "activation": "gelu",
        "revin": True,
        "num_members": 5,
    },
    "SOFTS": {
        "d_model": 512,
        "d_core": 512,
        "e_layers": 2,
        "dropout": 0.0,
        "revin": True,
    },
    "Koopa": {
        "seg_len": 10,
        "d_model": 128,
        "top_k": 5,
        "revin": True,
        "dropout": 0.0,
    },
    "LightTS": {
        "chunk_size": 8,
        "d_model": 64,
        "revin": True,
        "dropout": 0.0,
    },
    "CycleNet": {
        "cycle_len": 24,
        "backbone": "linear",
        "revin": True,
        "dropout": 0.0,
    },
    "WaveNet": {
        "d_model": 64,
        "d_skip": 64,
        "kernel_size": 2,
        "num_layers": 8,
        "num_stacks": 2,
        "dropout": 0.0,
        "revin": True,
    },
    "ETSformer": {
        "d_model": 256,
        "n_heads": 4,
        "e_layers": 2,
        "d_ff": 512,
        "dropout": 0.1,
        "top_k": 5,
        "revin": True,
    },
    "NSTransformer": {
        "d_model": 256,
        "n_heads": 8,
        "e_layers": 3,
        "d_ff": 512,
        "dropout": 0.1,
    },
    "MICN": {
        "d_model": 64,
        "num_scales": 3,
        "kernel_size": 5,
        "dropout": 0.05,
        "revin": True,
    },
    "TFT": {
        "d_model": 128,
        "n_heads": 4,
        "num_lstm_layers": 2,
        "dropout": 0.1,
    },
    "FiLM": {
        "d_order": 32,
        "n_lowpass": 2,
        "d_ff": 256,
        "dropout": 0.05,
        "revin": True,
    },
    "DishTS": {
        "d_model": 256,
        "n_heads": 8,
        "e_layers": 3,
        "d_ff": 512,
        "dropout": 0.1,
        "dish_hidden": 64,
    },
    "MambaForecaster": {
        "d_model": 64,
        "d_state": 16,
        "e_layers": 2,
        "dropout": 0.05,
        "revin": True,
    },
    "ModernTCN": {
        "patch_size": 8,
        "patch_stride": 4,
        "d_model": 128,
        "kernel_size": 51,
        "e_layers": 3,
        "d_ff_ratio": 4,
        "dropout": 0.05,
        "revin": True,
    },
    "RLinear": {
        "individual": True,
    },
    "FilterNet": {
        "num_filters": 8,
        "revin": True,
    },
    "CARD": {
        "d_model": 128,
        "n_heads": 4,
        "e_layers": 2,
        "d_ff": 256,
        "patch_len": 16,
        "stride": 8,
        "dropout": 0.1,
        "revin": True,
    },
    "Pathformer": {
        "patch_sizes": [4, 8, 16],
        "d_model": 64,
        "n_heads": 4,
        "e_layers": 2,
        "d_ff": 128,
        "dropout": 0.1,
        "revin": True,
    },
    "SMamba": {
        "d_model": 64,
        "d_state": 16,
        "e_layers": 2,
        "n_heads": 4,
        "d_ff": 128,
        "patch_len": 16,
        "stride": 16,
        "dropout": 0.05,
        "revin": True,
    },
    "iMamba": {
        "d_model": 128,
        "d_state": 16,
        "e_layers": 3,
        "d_ff": 256,
        "dropout": 0.05,
        "revin": True,
    },
    "Basisformer": {
        "n_basis": 32,
        "d_model": 64,
        "n_heads": 4,
        "e_layers": 2,
        "d_ff": 128,
        "dropout": 0.1,
        "revin": True,
    },
    "GCNForecaster": {
        "d_model": 64,
        "e_layers": 3,
        "d_emb": 10,
        "k_hops": 2,
        "kernel_size": 3,
        "dropout": 0.1,
        "revin": True,
    },
    "MoEForecaster": {
        "n_experts": 8,
        "k_active": 2,
        "d_router": 32,
        "expert_type": "linear",
        "d_ff": 128,
        "dropout": 0.1,
        "revin": True,
    },
    "RetForecaster": {
        "d_model": 64,
        "n_heads": 4,
        "e_layers": 2,
        "d_ff": 128,
        "patch_len": 16,
        "stride": 16,
        "dropout": 0.1,
        "revin": True,
    },
    "HarmonicForecaster": {
        "n_harmonics": 16,
        "use_mlp": True,
        "d_mlp": 64,
        "dropout": 0.1,
        "revin": True,
    },
    "HDMixer": {
        "patch_sizes": [4, 8, 16],
        "d_model": 64,
        "dropout": 0.1,
        "revin": True,
    },
    "KANForecaster": {
        "hidden": 64,
        "e_layers": 2,
        "degree": 5,
        "dropout": 0.1,
        "revin": True,
    },
    "TSReservoir": {
        "d_res": 256,
        "spectral_radius": 0.9,
        "input_scale": 0.1,
        "pool_states": True,
        "revin": True,
    },
    "WaveletForecaster": {
        "n_levels": 3,
        "revin": True,
    },
    "LinearAttentionForecaster": {
        "d_model": 64,
        "n_heads": 4,
        "e_layers": 2,
        "d_ff": 128,
        "patch_len": 16,
        "stride": 16,
        "dropout": 0.1,
        "revin": True,
    },
    "DualDecompForecaster": {
        "kernel_size": 25,
        "d_model": 64,
        "n_heads": 4,
        "e_layers": 2,
        "d_ff": 128,
        "patch_len": 8,
        "dropout": 0.1,
        "revin": True,
    },
    "HyperForecaster": {
        "d_ctx": 64,
        "hidden": 32,
        "d_ctx_hidden": 128,
        "dropout": 0.1,
        "revin": True,
    },
    "GATForecaster": {
        "d_model": 64,
        "n_heads": 4,
        "e_layers": 2,
        "d_ff": 128,
        "dropout": 0.1,
        "revin": True,
    },
    "BiLSTMForecaster": {
        "d_model": 64,
        "num_layers": 2,
        "d_attn": 32,
        "dropout": 0.1,
        "revin": True,
    },
    "RandomFourierForecaster": {
        "d_rff": 256,
        "sigma": 1.0,
        "revin": True,
    },
    "SparseTransformerForecaster": {
        "patch_size": 8,
        "d_model": 64,
        "n_heads": 4,
        "e_layers": 2,
        "d_ff": 128,
        "local_window": 3,
        "stride": 4,
        "dropout": 0.1,
        "revin": True,
    },
    "FourierMixerForecaster": {
        "e_layers": 3,
        "dropout": 0.1,
        "revin": True,
    },
    "TemporalConvAttentionForecaster": {
        "d_model": 64,
        "n_heads": 4,
        "n_blocks": 4,
        "kernel_size": 3,
        "dropout": 0.1,
        "revin": True,
    },
    "AdaptiveSpectralForecaster": {
        "n_filters": 16,
        "revin": True,
    },
    "LRUForecaster": {
        "d_model": 64,
        "d_state": 64,
        "n_layers": 3,
        "mlp_mult": 2,
        "dropout": 0.1,
        "revin": True,
    },
    "S4Forecaster": {
        "d_model": 64,
        "d_state": 32,
        "n_layers": 3,
        "mlp_mult": 2,
        "dropout": 0.1,
        "revin": True,
    },
    "HyenaForecaster": {
        "d_model": 64,
        "n_layers": 3,
        "pos_freqs": 16,
        "filter_dim": 64,
        "dropout": 0.1,
        "revin": True,
    },
    "PrototypicalForecaster": {
        "n_proto": 32,
        "d_proto": 64,
        "query_dim": 128,
        "dropout": 0.1,
        "revin": True,
    },
    "MultiscaleConvForecaster": {
        "d_model": 64,
        "n_layers": 3,
        "kernels": (3, 7, 15, 31),
        "dropout": 0.1,
        "revin": True,
    },
    "NeuralBasisForecaster": {
        "n_basis": 32,
        "d_hidden": 64,
        "dropout": 0.1,
        "revin": True,
    },
    "SincNetForecaster": {
        "n_filters": 32,
        "kernel_size": 25,
        "n_conv_layers": 2,
        "dropout": 0.1,
        "revin": True,
    },
    "GatedMLPForecaster": {
        "d_model": 64,
        "d_ffn": 128,
        "n_layers": 3,
        "dropout": 0.1,
        "revin": True,
    },
    "ImplicitNeuralForecaster": {
        "d_latent": 64,
        "d_time": 33,
        "enc_layers": 2,
        "dec_layers": 3,
        "d_hidden": 128,
        "dropout": 0.1,
        "revin": True,
    },
    "LiquidNetForecaster": {
        "d_model": 64,
        "n_layers": 2,
        "dt": 0.1,
        "dropout": 0.1,
        "revin": True,
    },
    "NormalizingFlow": {
        "d_model": 256,
        "n_heads": 4,
        "e_layers": 2,
        "d_ff": 512,
        "dropout": 0.1,
        "activation": "gelu",
        "revin": True,
        "num_samples": 50,
        "flow_layers": 6,
        "flow_hidden": 128,
    },
    "FreTS": {
        "channel_independence": True,
    },
    "FITS": {
        "individual": True,
        "cut_freq": 8,
    },
    "CATS": {
        "d_model": 256,
        "e_layers": 3,
        "d_layers": 3,
        "d_ff": 512,
        "dropout": 0.0,
        "n_heads": 8,
        "patch_len": 16,
        "stride": 8,
        "label_len": 48,
        "QAM_start": 0.1,
        "QAM_end": 0.3,
        "query_independence": True,
        "store_attn": True,
        "padding_patch": "end",
    },
    "DLinear": {"individual": False},
    "TCN": {
        "d_model": 64,
        "num_levels": 4,
        "kernel_size": 3,
        "dropout": 0.1,
        "revin": True,
    },
    "PatchMixer": {
        "patch_len": 16,
        "stride": 8,
        "d_model": 64,
        "depth": 3,
        "dropout": 0.1,
        "revin": True,
    },
    "RNN": {
        "hidden_size": 64,
        "num_layers": 2,
        "rnn_type": "gru",
        "dropout": 0.1,
        "bidirectional": False,
        "revin": True,
    },
    "FEDformer": {
        "d_ff": 2048,
        "d_model": 512,
        "embed": "timeF",
        "dropout": 0.0,
        "cross_activation": "tanh",
        "activation": "gelu",
        "version": "Fourier",
        "n_heads": 8,
        "L": 3,
        "moving_avg": 25,
        "e_layers": 2,
        "d_layers": 1,
        "modes": 64,
        "base": "legendre",
        "mode_select": "random",
    },
    "iTransformer": {
        "factor": 1,
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 2,
        "d_ff": 2048,
        "dropout": 0.1,
        "embed": "timeF",
        "activation": "gelu",
        "use_norm": True,
        "class_strategy": "projection",
    },
    "NLinear": {"individual": False},
    "PatchTST": {
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 1,
        "d_ff": 2048,
        "dropout": 0.1,
        "patch_len": 16,
        "stride": 8,
    },
    "TimesNet": {
        "n_heads": 8,
        "e_layers": 2,
        "label_len": 48,
        "d_model": 512,
        "d_ff": 512,
        "num_kernels": 6,
        "top_k": 5,
        "dropout": 0.0,
        "embed": "timeF",
    },
}


def _lr_for(model: str) -> float:
    return 0.001 if model in {"DLinear", "NLinear"} else 0.0001


def _training_params(model: str, task_name: str) -> dict:
    batch_size = 32
    if task_name == "anomaly_detection":
        batch_size = 128
    elif task_name == "uea_classification":
        batch_size = 16

    return {
        "epochs": 10,
        "patience": 3,
        "batch_size": batch_size,
        "lr": _lr_for(model),
        "max_grad_norm": None,
    }


def params_for(model: str, task_name: str) -> dict:
    params = deepcopy(MODEL_PARAMS[model])
    params.update(_training_params(model, task_name))

    if task_name in {"long_term_forecast", "short_term_forecast"}:
        params.update({
            "windows": 96,
            "horizon": 1,
            "time_enc": 1,
            "lradj": "type1",
            "experiment_label": ""
            if task_name == "long_term_forecast"
            else "short_term_forecast",
        })
    elif task_name == "imputation":
        params.update({"windows": 96})
    elif task_name == "anomaly_detection":
        params.update({"windows": 96, "spacing": 100, "anomaly_ratio": 0.25})
    elif task_name == "uea_classification":
        params.update({"windows": 96})
    elif task_name == "irregular_classification":
        pass  # no extra task params needed — dataset drives shape
    elif task_name == "irregular_interpolation":
        params.update({"query_rate": 0.2})
    elif task_name == "irregular_forecast":
        params.update({"obs_frac": 0.7})
    else:
        raise ValueError(f"Unknown reproduce task: {task_name}")

    return params
