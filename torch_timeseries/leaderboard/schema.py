from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


SUPPORTED_TASKS = (
    "Forecast",
    "Imputation",
    "AnomalyDetection",
    "UEAClassification",
)


@dataclass
class LeaderboardSource:
    source_type: str
    source_name: str
    citation: str = ""
    url: str = ""
    notes: str = ""


@dataclass
class LeaderboardEntry:
    model: str
    task: str
    dataset: str
    hparams: dict
    metrics: Dict[str, float]
    source: LeaderboardSource
    metric_mean: Dict[str, float] = field(default_factory=dict)
    metric_std: Dict[str, float] = field(default_factory=dict)
    num_seeds: int = 1
    seed: Optional[int] = None
    num_params: Optional[int] = None
    train_time_sec: Optional[float] = None
    git_commit: str = ""
    rank: Optional[int] = None

    def __post_init__(self):
        if not self.metric_mean:
            self.metric_mean = dict(self.metrics)
        if not self.metric_std:
            self.metric_std = {name: 0.0 for name in self.metric_mean}

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class LeaderboardTable:
    entries: List[LeaderboardEntry]

    def as_dicts(self) -> List[dict]:
        return [entry.as_dict() for entry in self.entries]
