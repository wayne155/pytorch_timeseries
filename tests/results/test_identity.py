from torch_timeseries.results.identity import (
    build_run_config,
    config_fingerprint,
    run_id_for_seed,
)


def test_run_config_fingerprint_is_stable_and_ignores_infra_hparams():
    run_config = build_run_config(
        model="DLinear",
        task="Forecast",
        dataset="ETTh1",
        hparams={
            "windows": 96,
            "pred_len": 96,
            "lr": 0.001,
            "scaler_type": "StandardScaler",
            "save_dir": "./results-a",
            "device": "cuda",
            "num_worker": 4,
            "pin_memory": True,
            "time_enc": 1,
        },
    )
    same_effective_config = build_run_config(
        model="DLinear",
        task="Forecast",
        dataset="ETTh1",
        hparams={
            "pred_len": 96,
            "windows": 96,
            "lr": 0.001,
            "scaler_type": "StandardScaler",
            "save_dir": "./results-b",
            "device": "cpu",
            "num_worker": 0,
            "pin_memory": False,
            "time_enc": 0,
        },
    )
    different_config = build_run_config(
        model="DLinear",
        task="Forecast",
        dataset="ETTh1",
        hparams={
            "windows": 192,
            "pred_len": 96,
            "lr": 0.001,
            "scaler_type": "StandardScaler",
        },
    )

    assert run_config == same_effective_config
    assert "save_dir" not in run_config["hparams"]
    assert "device" not in run_config["hparams"]
    assert "time_enc" not in run_config["hparams"]
    assert config_fingerprint(run_config) == config_fingerprint(same_effective_config)
    assert config_fingerprint(run_config) != config_fingerprint(different_config)
    assert run_id_for_seed(seed=3, config_hash="abc123") == "seed3-abc123"
