from __future__ import annotations

import datetime
import time
from dataclasses import asdict
from typing import Dict, List, Optional

from .experiments.configs import split_experiment_config
from .experiments.engine import CrossformerForecastEngine, DLinearForecastEngine
from .results.backends import (
    ArtifactBackend,
    LocalArtifactBackend,
    LocalBackend,
    ResultBackend,
    WandbBackend,
    _get_git_commit,
)
from .results.identity import build_run_config, config_fingerprint, run_id_for_seed
from .results.schema import RunResult


class Experiment:
    """Fluent builder for running and comparing time-series experiments."""

    def __init__(self, model: str, task: str, dataset: str, **kwargs) -> None:
        self.model = model
        self.task = task
        self.dataset = dataset
        self._overrides: Dict = dict(kwargs)
        self._backends: List[ResultBackend] = []
        self._artifact_backends: List[ArtifactBackend] = []
        save_dir = self._overrides.get("save_dir")
        if save_dir is not None:
            self._backends.append(LocalBackend(save_dir=save_dir))
            self._artifact_backends.append(LocalArtifactBackend(save_dir=save_dir))

    def set(self, **kwargs) -> "Experiment":
        self._overrides.update(kwargs)
        return self

    def with_local(self, save_dir: str = "./results") -> "Experiment":
        self._backends.append(LocalBackend(save_dir=save_dir))
        self._artifact_backends.append(LocalArtifactBackend(save_dir=save_dir))
        return self

    def with_wandb(self, project: str, entity: str = None) -> "Experiment":
        self._backends.append(WandbBackend(project=project, entity=entity))
        return self

    def _uses_engine_path(self) -> bool:
        return (self.model, self.task) in self._engine_classes()

    def _engine_classes(self):
        return {
            ("DLinear", "Forecast"): DLinearForecastEngine,
            ("Crossformer", "Forecast"): CrossformerForecastEngine,
        }

    def _identity_for(self, hparams: dict, seed: int):
        run_config = build_run_config(
            model=self.model,
            task=self.task,
            dataset=self.dataset,
            hparams=hparams,
        )
        config_hash = config_fingerprint(run_config)
        return run_config, config_hash, run_id_for_seed(seed, config_hash)

    def _save_model_artifacts(self, result: RunResult, source_path: str) -> dict:
        artifacts = {}
        for backend in self._artifact_backends:
            artifacts.update(backend.save_model(result, source_path))
        return artifacts

    def _run_engine_path(self, seed: int) -> RunResult:
        task_cfg, model_cfg, runtime_cfg = split_experiment_config(
            model=self.model,
            task=self.task,
            kwargs=self._overrides,
        )
        engine_cls = self._engine_classes()[(self.model, self.task)]
        hparams = {}
        hparams.update(asdict(task_cfg))
        hparams.update(asdict(model_cfg))
        hparams.update(asdict(runtime_cfg))
        run_config, config_hash, run_id = self._identity_for(hparams, seed)
        engine = engine_cls(
            model_name=self.model,
            dataset_name=self.dataset,
            task_config=task_cfg,
            model_config=model_cfg,
            runtime_config=runtime_cfg,
        )
        engine.run_config = run_config
        engine.config_hash = config_hash
        engine.run_id = run_id
        t0 = time.time()
        metrics = engine.run(seed)
        elapsed = time.time() - t0
        result = RunResult(
            model=self.model,
            task=self.task,
            dataset=self.dataset,
            seed=seed,
            timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
            hparams=engine.hparams(),
            metrics=metrics or {},
            num_params=engine.num_parameters(),
            train_time_sec=round(elapsed, 2),
            git_commit=_get_git_commit(),
            history=getattr(engine, "history", None),
            run_config=run_config,
            config_hash=config_hash,
            run_id=run_id,
        )
        artifacts = self._save_model_artifacts(
            result,
            getattr(engine, "best_checkpoint_filepath", ""),
        )
        result.artifacts = artifacts or None
        return result

    def run(self, seeds: List[int] = None) -> List[RunResult]:
        from torch_timeseries.experiments import get_experiment_class
        from torch_timeseries.utils.model_stats import count_parameters

        if seeds is None:
            seeds = [42]

        if self._uses_engine_path():
            results = []
            for seed in seeds:
                r = self._run_engine_path(seed)
                for backend in self._backends:
                    backend.save(r)
                results.append(r)
            return results

        exp_cls = get_experiment_class(self.model, self.task)

        results = []
        for seed in seeds:
            exp = exp_cls(dataset_type=self.dataset, **self._overrides)

            t0 = time.time()
            metrics = exp.run(seed)
            elapsed = time.time() - t0

            try:
                _, num_params = count_parameters(exp.model)
            except Exception:
                num_params = 0

            hparams = {}
            try:
                hparams = {k: v for k, v in asdict(exp).items()
                           if isinstance(v, (int, float, str, bool))}
            except Exception:
                pass
            run_config, config_hash, run_id = self._identity_for(hparams, seed)

            r = RunResult(
                model=self.model,
                task=self.task,
                dataset=self.dataset,
                seed=seed,
                timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
                hparams=hparams,
                metrics=metrics or {},
                num_params=num_params,
                train_time_sec=round(elapsed, 2),
                git_commit=_get_git_commit(),
                run_config=run_config,
                config_hash=config_hash,
                run_id=run_id,
            )
            artifacts = self._save_model_artifacts(
                r,
                getattr(exp, "best_checkpoint_filepath", ""),
            )
            r.artifacts = artifacts or None

            for backend in self._backends:
                backend.save(r)

            results.append(r)

        return results

    @classmethod
    def grid(
        cls,
        models: List[str],
        tasks: List[str],
        datasets: List[str],
        seeds: List[int] = None,
        save_dir: str = "./results",
        **shared_kwargs,
    ) -> "_GridRunner":
        return _GridRunner(
            models=models, tasks=tasks, datasets=datasets,
            seeds=seeds or [42], save_dir=save_dir,
            shared_kwargs=shared_kwargs,
        )

    @classmethod
    def compare(
        cls,
        save_dir: str = "./results",
        task: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> None:
        backend = LocalBackend(save_dir=save_dir)
        filters = {}
        if task:
            filters["task"] = task
        if dataset:
            filters["dataset"] = dataset
        results = backend.load_all(**filters)
        _print_comparison_table(results)


class _GridRunner:
    def __init__(self, models, tasks, datasets, seeds, save_dir, shared_kwargs):
        self._models = models
        self._tasks = tasks
        self._datasets = datasets
        self._seeds = seeds
        self._save_dir = save_dir
        self._shared_kwargs = shared_kwargs

    def run(self) -> List[RunResult]:
        all_results = []
        for model in self._models:
            for task in self._tasks:
                for dataset in self._datasets:
                    results = (
                        Experiment(model=model, task=task, dataset=dataset)
                        .set(**self._shared_kwargs)
                        .with_local(save_dir=self._save_dir)
                        .run(seeds=self._seeds)
                    )
                    all_results.extend(results)
        return all_results


def _print_comparison_table(results: List[RunResult]) -> None:
    if not results:
        print("No results found.")
        return

    from collections import defaultdict
    import statistics

    groups: dict = defaultdict(list)
    for r in results:
        groups[(r.task, r.dataset)].append(r)

    for (task, dataset), group in sorted(groups.items()):
        print(f"\nTask: {task} | Dataset: {dataset}")
        print("-" * 60)

        metric_keys = list(group[0].metrics.keys())
        header = f"{'Model':<20}" + "".join(f"{k:>20}" for k in metric_keys) + f"{'#params':>12}"
        print(header)

        by_model: dict = defaultdict(list)
        for r in group:
            by_model[r.model].append(r)

        for model, runs in sorted(by_model.items()):
            row = f"{model:<20}"
            for k in metric_keys:
                vals = [r.metrics[k] for r in runs if k in r.metrics]
                if vals:
                    mean = statistics.mean(vals)
                    if len(vals) > 1:
                        std = statistics.stdev(vals)
                        row += f"{f'{mean:.3f}+/-{std:.3f}':>20}"
                    else:
                        row += f"{f'{mean:.3f}':>20}"
                else:
                    row += f"{'N/A':>20}"
            params = runs[0].num_params
            row += f"{params:>12,}"
            print(row)


def register_model(model_cls) -> None:
    """Add a model class to the experiment registry for all supported tasks."""
    from dataclasses import dataclass
    from torch_timeseries.experiments import EXPERIMENT_REGISTRY

    # Lazy import to avoid circular
    from torch_timeseries.experiments.forecast import ForecastExp
    from torch_timeseries.experiments.imputation import ImputationExp
    from torch_timeseries.experiments.anomaly_detection import AnomalyDetectionExp
    from torch_timeseries.experiments.uea_classification import UEAClassificationExp

    task_map = {
        "Forecast": ForecastExp,
        "Imputation": ImputationExp,
        "AnomalyDetection": AnomalyDetectionExp,
        "UEAClassification": UEAClassificationExp,
    }

    model_name = model_cls.__name__
    for task_suffix, task_base in task_map.items():
        if issubclass(model_cls, task_base):
            # model_cls already inherits this task base — register as-is
            EXPERIMENT_REGISTRY[(model_name, task_suffix)] = model_cls
        else:
            combo_name = f"{model_name}{task_suffix}"
            combo_cls = dataclass(type(combo_name, (task_base, model_cls), {}))
            EXPERIMENT_REGISTRY[(model_name, task_suffix)] = combo_cls
