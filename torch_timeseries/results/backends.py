from __future__ import annotations

import json
import os
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
        return os.path.join(
            self.save_dir,
            f"{r.model}_{r.task}_{r.dataset}_seed{r.seed}.json",
        )

    def save(self, result: RunResult) -> None:
        with open(self._filename(result), "w") as f:
            json.dump(asdict(result), f, indent=2)

    def load_all(self, **filters) -> List[RunResult]:
        results = []
        for fname in os.listdir(self.save_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(self.save_dir, fname)) as f:
                d = json.load(f)
            r = RunResult.from_dict(d)
            if all(getattr(r, k, None) == v for k, v in filters.items()):
                results.append(r)
        return results


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
