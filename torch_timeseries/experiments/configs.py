from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Optional, Tuple, Union

from torch_timeseries.dataloader.v2 import SplitConfig, TimeEncConfig, WindowConfig


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
class TCNConfig:
    d_model: int = 64
    num_levels: int = 4
    kernel_size: int = 3
    dropout: float = 0.1
    revin: bool = True

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_levels <= 0:
            raise ValueError("num_levels must be positive")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class PatchMixerConfig:
    patch_len: int = 16
    patch_stride: int = 8
    d_model: int = 64
    depth: int = 3
    dropout: float = 0.1
    revin: bool = True

    def validate(self) -> None:
        if self.patch_len <= 0:
            raise ValueError("patch_len must be positive")
        if self.patch_stride <= 0:
            raise ValueError("patch_stride must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class RNNConfig:
    hidden_size: int = 64
    num_layers: int = 2
    rnn_type: str = "gru"
    dropout: float = 0.1
    bidirectional: bool = False
    revin: bool = True

    def validate(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.rnn_type not in ("gru", "lstm", "rnn"):
            raise ValueError("rnn_type must be one of 'gru', 'lstm', 'rnn'")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class VanillaTransformerConfig:
    d_model: int = 256
    n_heads: int = 4
    e_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    revin: bool = True

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.e_layers <= 0:
            raise ValueError("e_layers must be positive")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")
        if self.activation not in ("relu", "gelu"):
            raise ValueError("activation must be 'relu' or 'gelu'")


@dataclass
class GaussianConfig:
    d_model: int = 256
    n_heads: int = 4
    e_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    revin: bool = True
    num_samples: int = 100
    min_log_sigma: float = -10.0
    max_log_sigma: float = 2.0

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.e_layers <= 0:
            raise ValueError("e_layers must be positive")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")
        if self.activation not in ("relu", "gelu"):
            raise ValueError("activation must be 'relu' or 'gelu'")
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if self.min_log_sigma >= self.max_log_sigma:
            raise ValueError("min_log_sigma must be less than max_log_sigma")


@dataclass
class MCDropoutConfig:
    d_model: int = 256
    n_heads: int = 4
    e_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    revin: bool = True
    num_samples: int = 100

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.e_layers <= 0:
            raise ValueError("e_layers must be positive")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")
        if self.activation not in ("relu", "gelu"):
            raise ValueError("activation must be 'relu' or 'gelu'")
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")


@dataclass
class StudentTConfig:
    d_model: int = 256
    n_heads: int = 4
    e_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    revin: bool = True
    num_samples: int = 100
    min_log_sigma: float = -10.0
    max_log_sigma: float = 2.0
    min_log_nu: float = 0.69
    max_log_nu: float = 3.5

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.e_layers <= 0:
            raise ValueError("e_layers must be positive")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")
        if self.activation not in ("relu", "gelu"):
            raise ValueError("activation must be 'relu' or 'gelu'")
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if self.min_log_sigma >= self.max_log_sigma:
            raise ValueError("min_log_sigma must be less than max_log_sigma")
        if self.min_log_nu >= self.max_log_nu:
            raise ValueError("min_log_nu must be less than max_log_nu")


@dataclass
class NBEATSConfig:
    stack_types: list = None   # None → ["generic", "generic", "generic"]
    num_blocks: int = 3
    hidden_size: int = 256
    expansion_coefficient_dim: int = 32
    degree_of_polynomial: int = 3
    num_harmonics: int = 1

    def __post_init__(self):
        if self.stack_types is None:
            self.stack_types = ["generic", "generic", "generic"]

    def validate(self) -> None:
        valid = {"generic", "trend", "seasonality"}
        for s in self.stack_types:
            if s not in valid:
                raise ValueError(f"stack_types must be subsets of {valid}, got '{s}'")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.expansion_coefficient_dim <= 0:
            raise ValueError("expansion_coefficient_dim must be positive")
        if self.degree_of_polynomial < 0:
            raise ValueError("degree_of_polynomial must be non-negative")
        if self.num_harmonics < 1:
            raise ValueError("num_harmonics must be at least 1")


@dataclass
class SparseTSFConfig:
    period: int = None   # None → seq_len // 4 at runtime
    revin: bool = True

    def validate(self) -> None:
        if self.period is not None and self.period < 1:
            raise ValueError("period must be >= 1")


@dataclass
class EnsembleConfig:
    d_model: int = 256
    n_heads: int = 4
    e_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    revin: bool = True
    num_members: int = 5

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.e_layers <= 0:
            raise ValueError("e_layers must be positive")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")
        if self.activation not in ("relu", "gelu"):
            raise ValueError("activation must be 'relu' or 'gelu'")
        if self.num_members < 1:
            raise ValueError("num_members must be >= 1")


@dataclass
class WaveNetConfig:
    d_model: int = 64
    d_skip: int = 64
    kernel_size: int = 2
    num_layers: int = 8
    num_stacks: int = 1
    dropout: float = 0.0
    revin: bool = True

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.d_skip <= 0:
            raise ValueError("d_skip must be positive")
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be >= 2")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.num_stacks < 1:
            raise ValueError("num_stacks must be >= 1")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class CycleNetConfig:
    cycle_len: int = 24
    backbone: str = "linear"
    d_model: int = 512
    revin: bool = True
    dropout: float = 0.0

    def validate(self) -> None:
        if self.cycle_len < 1:
            raise ValueError("cycle_len must be >= 1")
        if self.backbone not in ("linear", "mlp"):
            raise ValueError("backbone must be 'linear' or 'mlp'")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class LightTSConfig:
    chunk_size: Optional[int] = None
    d_model: int = 64
    revin: bool = True
    dropout: float = 0.0

    def validate(self) -> None:
        if self.chunk_size is not None and self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1 or None")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class KoopaConfig:
    seg_len: int = 10
    d_model: int = 128
    n_ff: Optional[int] = None
    top_k: int = 5
    revin: bool = True
    dropout: float = 0.0

    def validate(self) -> None:
        if self.seg_len <= 0:
            raise ValueError("seg_len must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_ff is not None and self.n_ff <= 0:
            raise ValueError("n_ff must be positive or None")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class SOFTSConfig:
    d_model: int = 512
    d_core: Optional[int] = None
    e_layers: int = 2
    dropout: float = 0.0
    revin: bool = True

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.d_core is not None and self.d_core <= 0:
            raise ValueError("d_core must be positive or None")
        if self.e_layers <= 0:
            raise ValueError("e_layers must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class QuantileConfig:
    d_model: int = 256
    n_heads: int = 4
    e_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    revin: bool = True

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.e_layers <= 0:
            raise ValueError("e_layers must be positive")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")
        if self.activation not in ("relu", "gelu"):
            raise ValueError("activation must be 'relu' or 'gelu'")


@dataclass
class RuntimeConfig:
    data_path: str = "~/.torchtimeseries/data"
    save_dir: str = "./results"
    device: str = "cpu"
    scaler_type: str = "StandardScaler"
    optm_type: str = "Adam"
    loss_func_type: str = "mse"
    batch_size: int = 32
    num_worker: int = 0
    lr: float = 0.0001
    l2_weight_decay: float = 0.0
    # TSLib protocol: 10 epochs, patience 3, halve lr each epoch, no clipping.
    epochs: int = 10
    patience: int = 3
    lradj: str = "type1"
    """LR schedule: "type1" = halve every epoch (TSLib default), "cosine"."""
    max_grad_norm: Optional[float] = None
    """Gradient-norm clip threshold; None disables clipping (TSLib default)."""
    invtrans_loss: bool = False
    pin_memory: bool = False
    experiment_label: str = ""

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
        if self.lradj not in ("type1", "cosine"):
            raise ValueError("lradj must be 'type1' or 'cosine'")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or None")


def _field_names(cls) -> set:
    return {field.name for field in fields(cls)}


def _build(cls, values: Dict):
    names = _field_names(cls)
    return cls(**{key: values.pop(key) for key in list(values) if key in names})


def _build_window_split(remaining: Dict) -> Tuple[WindowConfig, Optional[SplitConfig]]:
    """Build (WindowConfig, SplitConfig|None) from nested configs or flat
    legacy keys (windows/pred_len/train_ratio/...). split=None means the
    dataset's default split."""
    window = remaining.pop("window", None)
    split = remaining.pop("split", None)

    flat = {
        k: remaining.pop(k)
        for k in (
            "windows", "pred_len", "horizon", "stride", "time_enc", "freq",
            "input_columns", "target_columns", "fast_val", "fast_test",
        )
        if k in remaining
    }
    ratios = {
        k: remaining.pop(k)
        for k in ("train_ratio", "val_ratio", "test_ratio")
        if k in remaining
    }

    if window is None:
        window = WindowConfig(
            window=flat.get("windows", 96),
            horizon=flat.get("horizon", 1),
            steps=flat.get("pred_len", 96),
            stride=flat.get("stride", 1),
            fast_val=flat.get("fast_val", False),
            fast_test=flat.get("fast_test", False),
            time_enc_cfg=TimeEncConfig(
                time_enc=flat.get("time_enc", 1),
                freq=flat.get("freq"),
            ),
            input_columns=flat.get("input_columns"),
            target_columns=flat.get("target_columns"),
        )
    elif flat:
        raise TypeError(
            f"pass either a WindowConfig or flat keys, not both: {sorted(flat)}"
        )

    if split is None:
        explicit = {k: v for k, v in ratios.items() if v is not None}
        if explicit:
            split = SplitConfig(
                train=explicit.get("train_ratio", 0.7),
                val=explicit.get("val_ratio"),
                test=explicit.get("test_ratio"),
            )
    elif any(v is not None for v in ratios.values()):
        raise TypeError("pass either a SplitConfig or flat ratio keys, not both")

    if window.window <= 0 or window.steps <= 0 or window.horizon <= 0 or window.stride <= 0:
        raise ValueError("window/pred_len/horizon/stride must all be positive")
    return window, split


def split_experiment_config(
    model: str,
    task: str,
    kwargs: Dict,
) -> Tuple[WindowConfig, Optional[SplitConfig], Union[DLinearConfig, CrossformerConfig], RuntimeConfig]:
    model_configs = {
        ("DLinear", "Forecast"): DLinearConfig,
        ("Crossformer", "Forecast"): CrossformerConfig,
        ("TCN", "Forecast"): TCNConfig,
        ("PatchMixer", "Forecast"): PatchMixerConfig,
        ("RNN", "Forecast"): RNNConfig,
        ("VanillaTransformer", "Forecast"): VanillaTransformerConfig,
        ("MCDropout", "Forecast"): MCDropoutConfig,
        ("Gaussian", "Forecast"): GaussianConfig,
        ("StudentT", "Forecast"): StudentTConfig,
        ("Quantile", "Forecast"): QuantileConfig,
        ("NBEATS", "Forecast"): NBEATSConfig,
        ("NBEATS", "AnomalyDetection"): NBEATSConfig,
        ("NBEATS", "Imputation"): NBEATSConfig,
        ("NBEATS", "UEAClassification"): NBEATSConfig,
        ("SparseTSF", "Forecast"): SparseTSFConfig,
        ("SparseTSF", "AnomalyDetection"): SparseTSFConfig,
        ("SparseTSF", "Imputation"): SparseTSFConfig,
        ("SparseTSF", "UEAClassification"): SparseTSFConfig,
        ("SOFTS", "Forecast"): SOFTSConfig,
        ("SOFTS", "AnomalyDetection"): SOFTSConfig,
        ("SOFTS", "Imputation"): SOFTSConfig,
        ("SOFTS", "UEAClassification"): SOFTSConfig,
        ("Koopa", "Forecast"): KoopaConfig,
        ("Koopa", "AnomalyDetection"): KoopaConfig,
        ("Koopa", "Imputation"): KoopaConfig,
        ("Koopa", "UEAClassification"): KoopaConfig,
        ("LightTS", "Forecast"): LightTSConfig,
        ("LightTS", "AnomalyDetection"): LightTSConfig,
        ("LightTS", "Imputation"): LightTSConfig,
        ("LightTS", "UEAClassification"): LightTSConfig,
        ("CycleNet", "Forecast"): CycleNetConfig,
        ("CycleNet", "AnomalyDetection"): CycleNetConfig,
        ("CycleNet", "Imputation"): CycleNetConfig,
        ("CycleNet", "UEAClassification"): CycleNetConfig,
        ("WaveNet", "Forecast"): WaveNetConfig,
        ("WaveNet", "AnomalyDetection"): WaveNetConfig,
        ("WaveNet", "Imputation"): WaveNetConfig,
        ("WaveNet", "UEAClassification"): WaveNetConfig,
        ("Ensemble", "Forecast"): EnsembleConfig,
    }
    if (model, task) not in model_configs:
        raise NotImplementedError(
            "typed config split is only implemented for migrated forecast models"
        )

    remaining = dict(kwargs)
    window_cfg, split_cfg = _build_window_split(remaining)
    model_cfg = _build(model_configs[(model, task)], remaining)
    runtime_cfg = _build(RuntimeConfig, remaining)

    if remaining:
        unknown = ", ".join(sorted(remaining))
        raise TypeError(f"Unknown or irrelevant configuration keys: {unknown}")

    model_cfg.validate()
    runtime_cfg.validate()
    return window_cfg, split_cfg, model_cfg, runtime_cfg
