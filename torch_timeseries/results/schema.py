from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class RunResult:
    model: str
    task: str
    dataset: str
    seed: int
    timestamp: str
    hparams: dict
    metrics: dict
    num_params: int
    train_time_sec: float
    git_commit: str
    history: Optional[dict] = None
    run_config: Optional[dict] = None
    config_hash: str = ""
    run_id: str = ""
    artifacts: Optional[dict] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        return cls(**d)
