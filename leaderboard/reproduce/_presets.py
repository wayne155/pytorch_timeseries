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
