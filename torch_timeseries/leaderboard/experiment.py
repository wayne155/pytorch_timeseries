"""LeaderboardExperiment — runs a model and writes
leaderboard/results/{Model}/{Task}/{Dataset}/w{W}_p{P}_seed{N}/metrics.json.

Models with a typed engine (DLinear, Crossformer) use the engine path;
everything else falls back to the EXPERIMENT_REGISTRY classes.
"""
from __future__ import annotations

import json
import pathlib
import re
import time
from dataclasses import asdict
from typing import Dict, List, Optional

from torch_timeseries.experiments.configs import split_experiment_config
from torch_timeseries.experiments.engine import DLinearForecastEngine, CrossformerForecastEngine

_ENGINE_MAP = {
    ("DLinear", "Forecast"): DLinearForecastEngine,
    ("Crossformer", "Forecast"): CrossformerForecastEngine,
}

# Infrastructure fields that say nothing about the experiment itself.
_HPARAM_EXCLUDE = {
    "data_path", "save_dir", "device",
    "model_type", "dataset_type",
}

_METRIC_ALIASES = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F-score": "f1",
}


def _clean_part(value) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


class LeaderboardExperiment:
    """Run a model experiment and persist results in the leaderboard directory layout.

    Output path: ``{results_dir}/{Model}/{Task}/{Dataset}/w{W}_p{P}_seed{N}/metrics.json``
    """

    def __init__(
        self,
        model: str,
        task: str,
        dataset: str,
        results_dir: str = "leaderboard/results",
        **kwargs,
    ) -> None:
        self.model = model
        self.task = task
        self.dataset = dataset
        self.results_dir = pathlib.Path(results_dir)
        self._kwargs = kwargs

    def _record_dir_name(self, seed: int, hparams: dict) -> str:
        parts = []
        if hparams.get("experiment_label"):
            parts.append(_clean_part(hparams["experiment_label"]))
        if "windows" in hparams:
            parts.append(f"w{_clean_part(hparams['windows'])}")
        elif "seq_len" in hparams:
            parts.append(f"seq{_clean_part(hparams['seq_len'])}")
        if "pred_len" in hparams:
            parts.append(f"p{_clean_part(hparams['pred_len'])}")
        if "mask_rate" in hparams:
            parts.append(f"m{_clean_part(hparams['mask_rate'])}")
        if "anomaly_ratio" in hparams:
            parts.append(f"ar{_clean_part(hparams['anomaly_ratio'])}")
        if not parts:
            parts.append("default")
        parts.append(f"seed{seed}")
        return "_".join(parts)

    def _write_record(self, seed: int, hparams: dict, metrics: Dict[str, float],
                      num_params: int, elapsed: float) -> dict:
        metrics = {_METRIC_ALIASES.get(k, k): v for k, v in (metrics or {}).items()}
        record = {
            "model": self.model,
            "task": self.task,
            "dataset": self.dataset,
            "seed": seed,
            "hparams": {k: v for k, v in hparams.items() if k not in _HPARAM_EXCLUDE},
            "metrics": metrics,
            "num_params": num_params,
            "train_time_sec": round(elapsed, 2),
        }
        pred_len = hparams.get("pred_len", "NA")
        out_dir = (
            self.results_dir / self.model / self.task / self.dataset
            / self._record_dir_name(seed, record["hparams"])
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(record, indent=2))
        print(f"[leaderboard] {self.model}/{self.dataset} pred_len={pred_len} "
              f"seed={seed} → " +
              " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items())) +
              f"  ({elapsed:.0f}s)")
        return record

    def _run_engine(self, seeds: List[int]) -> List[dict]:
        window_cfg, split_cfg, model_cfg, runtime_cfg = split_experiment_config(
            self.model, self.task, dict(self._kwargs)
        )
        engine_cls = _ENGINE_MAP[(self.model, self.task)]
        records = []
        for seed in seeds:
            engine = engine_cls(
                model_name=self.model,
                dataset_name=self.dataset,
                window_config=window_cfg,
                split_config=split_cfg,
                model_config=model_cfg,
                runtime_config=runtime_cfg,
            )
            t0 = time.time()
            metrics = engine.run(seed=seed)
            records.append(self._write_record(
                seed, engine.hparams(), metrics,
                engine.num_parameters(), time.time() - t0,
            ))
        return records

    def _run_registry(self, seeds: List[int]) -> List[dict]:
        from torch_timeseries.experiments import get_experiment_class
        from torch_timeseries.utils.model_stats import count_parameters

        exp_cls = get_experiment_class(self.model, self.task)
        records = []
        for seed in seeds:
            exp = exp_cls(dataset_type=self.dataset, **self._kwargs)
            t0 = time.time()
            metrics = exp.run(seed)
            elapsed = time.time() - t0
            try:
                _, num_params = count_parameters(exp.model)
            except Exception:
                num_params = 0
            hparams = {k: v for k, v in asdict(exp).items()
                       if isinstance(v, (int, float, str, bool))}
            records.append(self._write_record(
                seed, hparams, metrics or {}, num_params, elapsed,
            ))
        return records

    def run(self, seeds: Optional[List[int]] = None) -> List[Dict]:
        if seeds is None:
            seeds = [42]
        if (self.model, self.task) in _ENGINE_MAP:
            return self._run_engine(seeds)
        return self._run_registry(seeds)
