from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict


INFRA_HPARAM_KEYS = {
    "data_path",
    "device",
    "num_worker",
    "pin_memory",
    "save_dir",
}


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _normalize(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
            if val is not None
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def effective_hparams(model: str, task: str, hparams: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        key: value
        for key, value in hparams.items()
        if key not in INFRA_HPARAM_KEYS and value is not None
    }
    if model == "DLinear" and task == "Forecast":
        out.pop("time_enc", None)
    return _normalize(out)


def build_run_config(
    model: str,
    task: str,
    dataset: str,
    hparams: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "model": model,
        "task": task,
        "dataset": dataset,
        "hparams": effective_hparams(model, task, hparams),
    }


def config_fingerprint(run_config: Dict[str, Any], length: int = 16) -> str:
    payload = json.dumps(
        _normalize(run_config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def run_id_for_seed(seed: int, config_hash: str) -> str:
    return f"seed{seed}-{config_hash}"
