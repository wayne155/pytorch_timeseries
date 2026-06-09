from __future__ import annotations

import json
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import List, Optional

from .schema import RunResult


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


class ResultBackend(ABC):
    @abstractmethod
    def save(self, result: RunResult) -> None: ...

    @abstractmethod
    def load_all(self, **filters) -> List[RunResult]: ...


class LocalBackend(ResultBackend):
    def __init__(self, save_dir: str = "./results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _filename(self, r: RunResult) -> str:
        if r.config_hash:
            return os.path.join(
                self.save_dir,
                "records",
                r.model,
                r.task,
                r.dataset,
                r.config_hash,
                f"seed{r.seed}.json",
            )
        return os.path.join(
            self.save_dir,
            f"{r.model}_{r.task}_{r.dataset}_seed{r.seed}.json",
        )

    def save(self, result: RunResult) -> None:
        filename = self._filename(result)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if result.config_hash and result.run_config:
            config_path = os.path.join(os.path.dirname(filename), "config.json")
            with open(config_path, "w") as f:
                json.dump(
                    {
                        "config_hash": result.config_hash,
                        "run_config": result.run_config,
                    },
                    f,
                    indent=2,
                )
        with open(filename, "w") as f:
            json.dump(asdict(result), f, indent=2)

    def load_all(self, **filters) -> List[RunResult]:
        results = []
        for root, _, files in os.walk(self.save_dir):
            for fname in files:
                if not fname.endswith(".json") or fname == "config.json":
                    continue
                with open(os.path.join(root, fname)) as f:
                    d = json.load(f)
                try:
                    r = RunResult.from_dict(d)
                except TypeError:
                    continue
                if all(getattr(r, k, None) == v for k, v in filters.items()):
                    results.append(r)
        return results


class ArtifactBackend(ABC):
    @abstractmethod
    def save_model(self, result: RunResult, source_path: str) -> dict: ...


class LocalArtifactBackend(ArtifactBackend):
    def __init__(self, save_dir: str = "./results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _relative_model_path(self, result: RunResult) -> str:
        config_hash = result.config_hash or "unfingerprinted"
        return os.path.join(
            "artifacts",
            result.model,
            result.task,
            result.dataset,
            config_hash,
            f"seed{result.seed}",
            "best_model.pth",
        )

    def save_model(self, result: RunResult, source_path: str) -> dict:
        if not source_path or not os.path.exists(source_path):
            return {}
        rel_path = self._relative_model_path(result)
        dest_path = os.path.join(self.save_dir, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        if os.path.abspath(source_path) != os.path.abspath(dest_path):
            shutil.copy2(source_path, dest_path)
        return {"model": rel_path}


class WandbBackend(ResultBackend):
    """Save results to Weights & Biases. Requires ``pip install wandb``."""

    def __init__(self, project: str, entity: Optional[str] = None):
        try:
            import wandb as _w
            self._wandb = _w
        except ImportError:
            raise ImportError(
                "wandb is required for WandbBackend: pip install wandb"
            )
        self.project = project
        self.entity = entity

    def save(self, result: RunResult) -> None:
        run = self._wandb.init(
            project=self.project,
            entity=self.entity,
            name=f"{result.model}_{result.task}_{result.dataset}_seed{result.seed}",
            config={**result.hparams, "model": result.model, "task": result.task,
                    "dataset": result.dataset, "seed": result.seed},
            reinit=True,
        )
        self._wandb.log(result.metrics)
        self._wandb.run.summary.update({
            "num_params": result.num_params,
            "train_time_sec": result.train_time_sec,
            "git_commit": result.git_commit,
        })
        run.finish()

    def load_all(self, **filters) -> List[RunResult]:
        raise NotImplementedError("WandbBackend.load_all is not supported; use LocalBackend for reads.")
