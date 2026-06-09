from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class ForecastConfig:
    windows: int = 96
    pred_len: int = 96
    horizon: int = 1
    train_ratio: float = 0.7
    val_ratio: Optional[float] = None
    test_ratio: float = 0.2
    time_enc: int = 1
    input_columns: Optional[List[int]] = None
    target_columns: Optional[List[int]] = None
    stride: int = 1
    scale_in_train: bool = True

    def validate(self) -> None:
        if self.windows <= 0:
            raise ValueError("windows must be positive")
        if self.pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if not (0 < self.train_ratio < 1):
            raise ValueError("train_ratio must be between 0 and 1")
        if self.test_ratio is not None and not (0 < self.test_ratio < 1):
            raise ValueError("test_ratio must be between 0 and 1")
        if self.val_ratio is not None and not (0 <= self.val_ratio < 1):
            raise ValueError("val_ratio must be between 0 and 1")


@dataclass
class DLinearConfig:
    individual: bool = False

    def validate(self) -> None:
        if not isinstance(self.individual, bool):
            raise ValueError("individual must be a bool")


@dataclass
class CrossformerConfig:
    seg_len: int = 6
    win_size: int = 2
    factor: int = 10
    d_model: int = 256
    d_ff: int = 512
    n_heads: int = 4
    e_layers: int = 3
    dropout: float = 0.2
    baseline: bool = False

    def validate(self) -> None:
        if self.seg_len <= 0:
            raise ValueError("seg_len must be positive")
        if self.win_size <= 0:
            raise ValueError("win_size must be positive")
        if self.factor <= 0:
            raise ValueError("factor must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.e_layers <= 0:
            raise ValueError("e_layers must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")
        if not isinstance(self.baseline, bool):
            raise ValueError("baseline must be a bool")


@dataclass
class RuntimeConfig:
    data_path: str = "./data"
    save_dir: str = "./results"
    device: str = "cpu"
    scaler_type: str = "StandardScaler"
    optm_type: str = "Adam"
    loss_func_type: str = "mse"
    batch_size: int = 32
    num_worker: int = 0
    lr: float = 0.0001
    l2_weight_decay: float = 0.0
    epochs: int = 20
    patience: int = 5
    max_grad_norm: float = 5.0
    invtrans_loss: bool = False
    pin_memory: bool = False

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_worker < 0:
            raise ValueError("num_worker must be non-negative")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


def _field_names(cls) -> set:
    return {field.name for field in fields(cls)}


def _build(cls, values: Dict):
    names = _field_names(cls)
    return cls(**{key: values.pop(key) for key in list(values) if key in names})


def split_experiment_config(
    model: str,
    task: str,
    kwargs: Dict,
) -> Tuple[ForecastConfig, Union[DLinearConfig, CrossformerConfig], RuntimeConfig]:
    model_configs = {
        ("DLinear", "Forecast"): DLinearConfig,
        ("Crossformer", "Forecast"): CrossformerConfig,
    }
    if (model, task) not in model_configs:
        raise NotImplementedError(
            "typed config split is only implemented for migrated forecast models"
        )

    remaining = dict(kwargs)
    task_cfg = _build(ForecastConfig, remaining)
    model_cfg = _build(model_configs[(model, task)], remaining)
    runtime_cfg = _build(RuntimeConfig, remaining)

    if remaining:
        unknown = ", ".join(sorted(remaining))
        raise TypeError(f"Unknown or irrelevant configuration keys: {unknown}")

    task_cfg.validate()
    model_cfg.validate()
    runtime_cfg.validate()
    return task_cfg, model_cfg, runtime_cfg
