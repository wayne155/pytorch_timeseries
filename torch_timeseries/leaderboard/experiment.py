"""LeaderboardExperiment — wraps a forecast engine and writes
leaderboard/results/{Model}/{Task}/{Dataset}/seed{N}/metrics.json."""
from __future__ import annotations

import json
import pathlib
import time
from typing import Dict, List, Optional

from torch_timeseries.experiments.configs import split_experiment_config
from torch_timeseries.experiments.engine import DLinearForecastEngine, CrossformerForecastEngine

_ENGINE_MAP = {
    ("DLinear", "Forecast"): DLinearForecastEngine,
    ("Crossformer", "Forecast"): CrossformerForecastEngine,
}

_HPARAM_EXCLUDE = {"data_path", "save_dir"}


class LeaderboardExperiment:
    """Run a model experiment and persist results in the leaderboard directory layout.

    Output path: ``{results_dir}/{Model}/{Task}/{Dataset}/seed{N}/metrics.json``
    """

    def __init__(
        self,
        model: str,
        task: str,
        dataset: str,
        results_dir: str = "leaderboard/results",
        **kwargs,
    ) -> None:
        if (model, task) not in _ENGINE_MAP:
            raise NotImplementedError(
                f"No leaderboard engine registered for ({model!r}, {task!r}). "
                f"Available: {list(_ENGINE_MAP)}"
            )
        self.model = model
        self.task = task
        self.dataset = dataset
        self.results_dir = pathlib.Path(results_dir)
        self._kwargs = kwargs

    def run(self, seeds: Optional[List[int]] = None) -> List[Dict]:
        if seeds is None:
            seeds = [42]

        # Split kwargs into typed config objects (fresh copy per call).
        task_cfg, model_cfg, runtime_cfg = split_experiment_config(
            self.model, self.task, dict(self._kwargs)
        )
        engine_cls = _ENGINE_MAP[(self.model, self.task)]

        records = []
        for seed in seeds:
            engine = engine_cls(
                model_name=self.model,
                dataset_name=self.dataset,
                task_config=task_cfg,
                model_config=model_cfg,
                runtime_config=runtime_cfg,
            )
            t0 = time.time()
            metrics = engine.run(seed=seed)
            elapsed = round(time.time() - t0, 2)

            record = {
                "model": self.model,
                "task": self.task,
                "dataset": self.dataset,
                "seed": seed,
                "hparams": {k: v for k, v in engine.hparams().items()
                            if k not in _HPARAM_EXCLUDE},
                "metrics": metrics,
                "num_params": engine.num_parameters(),
                "train_time_sec": elapsed,
            }

            out_dir = (
                self.results_dir
                / self.model
                / self.task
                / self.dataset
                / f"w{task_cfg.windows}_p{task_cfg.pred_len}_seed{seed}"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "metrics.json").write_text(json.dumps(record, indent=2))
            print(f"[leaderboard] {self.model}/{self.dataset} pred_len={task_cfg.pred_len} "
                  f"seed={seed} → mse={metrics.get('mse', '?'):.4f} "
                  f"mae={metrics.get('mae', '?'):.4f}  ({elapsed:.0f}s)")
            records.append(record)

        return records
