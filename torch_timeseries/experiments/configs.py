from __future__ import annotations

from dataclasses import dataclass, field, fields
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
class ETSformerConfig:
    d_model: int = 256
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    top_k: int = 5
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
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")


@dataclass
class FilterNetConfig:
    num_filters: int = 8
    revin: bool = True

    def validate(self) -> None:
        if self.num_filters < 1:
            raise ValueError("num_filters must be >= 1")


@dataclass
class RetForecasterConfig:
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 128
    patch_len: int = 16
    stride: int = 16
    dropout: float = 0.1
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
        if self.patch_len <= 0:
            raise ValueError("patch_len must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class GATForecasterConfig:
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
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
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class BiLSTMForecasterConfig:
    d_model: int = 64
    num_layers: int = 2
    d_attn: int = 32
    dropout: float = 0.1
    revin: bool = True

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.d_attn <= 0:
            raise ValueError("d_attn must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class RandomFourierForecasterConfig:
    d_rff: int = 256
    sigma: float = 1.0
    revin: bool = True

    def validate(self) -> None:
        if self.d_rff <= 0:
            raise ValueError("d_rff must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")


@dataclass
class SparseTransformerForecasterConfig:
    patch_size: int = 8
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 128
    local_window: int = 3
    stride: int = 4
    dropout: float = 0.1
    revin: bool = True

    def validate(self) -> None:
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.e_layers <= 0:
            raise ValueError("e_layers must be positive")
        if self.local_window < 0:
            raise ValueError("local_window must be non-negative")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class FourierMixerForecasterConfig:
    e_layers: int = 3
    dropout: float = 0.1
    revin: bool = True

    def validate(self) -> None:
        if self.e_layers <= 0:
            raise ValueError("e_layers must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class TemporalConvAttentionForecasterConfig:
    d_model: int = 64
    n_heads: int = 4
    n_blocks: int = 4
    kernel_size: int = 3
    dropout: float = 0.1
    revin: bool = True

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_blocks <= 0:
            raise ValueError("n_blocks must be positive")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class AdaptiveSpectralForecasterConfig:
    n_filters: int = 16
    revin: bool = True

    def validate(self) -> None:
        if self.n_filters <= 0:
            raise ValueError("n_filters must be positive")


@dataclass
class MultiscaleConvForecasterConfig:
    d_model: int = 64
    n_layers: int = 3
    kernels: tuple = (3, 7, 15, 31)
    dropout: float = 0.1
    revin: bool = True

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.d_model % len(self.kernels) != 0:
            raise ValueError("d_model must be divisible by len(kernels)")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class PrototypicalForecasterConfig:
    n_proto: int = 32
    d_proto: int = 64
    query_dim: int = 128
    dropout: float = 0.1
    revin: bool = True

    def validate(self) -> None:
        if self.n_proto <= 0:
            raise ValueError("n_proto must be positive")
        if self.d_proto <= 0:
            raise ValueError("d_proto must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class HyenaForecasterConfig:
    d_model: int = 64
    n_layers: int = 3
    pos_freqs: int = 16
    filter_dim: int = 64
    dropout: float = 0.1
    revin: bool = True

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.pos_freqs <= 0:
            raise ValueError("pos_freqs must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class S4ForecasterConfig:
    d_model: int = 64
    d_state: int = 32
    n_layers: int = 3
    mlp_mult: int = 2
    dropout: float = 0.1
    revin: bool = True

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.d_state <= 0:
            raise ValueError("d_state must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class LRUForecasterConfig:
    d_model: int = 64
    d_state: int = 64
    n_layers: int = 3
    mlp_mult: int = 2
    dropout: float = 0.1
    revin: bool = True

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.d_state <= 0:
            raise ValueError("d_state must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class HyperForecasterConfig:
    d_ctx: int = 64
    hidden: int = 32
    d_ctx_hidden: int = 128
    dropout: float = 0.1
    revin: bool = True

    def validate(self) -> None:
        if self.d_ctx <= 0:
            raise ValueError("d_ctx must be positive")
        if self.hidden <= 0:
            raise ValueError("hidden must be positive")
        if self.d_ctx_hidden <= 0:
            raise ValueError("d_ctx_hidden must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class DualDecompForecasterConfig:
    kernel_size: int = 25
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 128
    patch_len: int = 8
    dropout: float = 0.1
    revin: bool = True

    def validate(self) -> None:
        if self.kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class LinearAttentionForecasterConfig:
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 128
    patch_len: int = 16
    stride: int = 16
    dropout: float = 0.1
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
        if self.patch_len <= 0:
            raise ValueError("patch_len must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class WaveletForecasterConfig:
    n_levels: int = 3
    revin: bool = True

    def validate(self) -> None:
        if self.n_levels < 1:
            raise ValueError("n_levels must be >= 1")


@dataclass
class TSReservoirConfig:
    d_res: int = 256
    spectral_radius: float = 0.9
    input_scale: float = 0.1
    pool_states: bool = True
    revin: bool = True

    def validate(self) -> None:
        if self.d_res <= 0:
            raise ValueError("d_res must be positive")
        if not (0 < self.spectral_radius < 1):
            raise ValueError("spectral_radius must be in (0, 1)")
        if self.input_scale <= 0:
            raise ValueError("input_scale must be positive")


@dataclass
class KANForecasterConfig:
    hidden: int = 64
    e_layers: int = 2
    degree: int = 5
    dropout: float = 0.1
    revin: bool = True

    def validate(self) -> None:
        if self.hidden <= 0:
            raise ValueError("hidden must be positive")
        if self.e_layers <= 0:
            raise ValueError("e_layers must be positive")
        if self.degree < 1:
            raise ValueError("degree must be >= 1")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class HDMixerConfig:
    patch_sizes: list = field(default_factory=lambda: [4, 8, 16])
    d_model: int = 64
    dropout: float = 0.1
    revin: bool = True

    def validate(self) -> None:
        if not self.patch_sizes:
            raise ValueError("patch_sizes must be non-empty")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class HarmonicForecasterConfig:
    n_harmonics: int = 16
    use_mlp: bool = True
    d_mlp: int = 64
    dropout: float = 0.1
    revin: bool = True

    def validate(self) -> None:
        if self.n_harmonics <= 0:
            raise ValueError("n_harmonics must be positive")
        if self.d_mlp <= 0:
            raise ValueError("d_mlp must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class MoEForecasterConfig:
    n_experts: int = 8
    k_active: int = 2
    d_router: int = 32
    expert_type: str = "linear"
    d_ff: int = 128
    dropout: float = 0.1
    revin: bool = True

    def validate(self) -> None:
        if self.n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        if self.k_active < 1:
            raise ValueError("k_active must be >= 1")
        if self.k_active > self.n_experts:
            raise ValueError("k_active must be <= n_experts")
        if self.expert_type not in ("linear", "mlp"):
            raise ValueError("expert_type must be 'linear' or 'mlp'")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class GCNForecasterConfig:
    d_model: int = 64
    e_layers: int = 3
    d_emb: int = 10
    k_hops: int = 2
    kernel_size: int = 3
    dropout: float = 0.1
    revin: bool = True

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.e_layers <= 0:
            raise ValueError("e_layers must be positive")
        if self.d_emb <= 0:
            raise ValueError("d_emb must be positive")
        if self.k_hops < 0:
            raise ValueError("k_hops must be >= 0")
        if self.kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class BasisformerConfig:
    n_basis: int = 32
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    revin: bool = True

    def validate(self) -> None:
        if self.n_basis < 1:
            raise ValueError("n_basis must be >= 1")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.e_layers <= 0:
            raise ValueError("e_layers must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class iMambaConfig:
    d_model: int = 128
    d_state: int = 16
    e_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.05
    revin: bool = True

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.d_state <= 0:
            raise ValueError("d_state must be positive")
        if self.e_layers <= 0:
            raise ValueError("e_layers must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class SMambaConfig:
    d_model: int = 64
    d_state: int = 16
    e_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    patch_len: int = 16
    stride: int = 16
    dropout: float = 0.1
    revin: bool = True

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.d_state <= 0:
            raise ValueError("d_state must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.e_layers <= 0:
            raise ValueError("e_layers must be positive")
        if self.patch_len <= 0:
            raise ValueError("patch_len must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class PathformerConfig:
    patch_sizes: list = field(default_factory=lambda: [4, 8, 16])
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    revin: bool = True

    def validate(self) -> None:
        if len(self.patch_sizes) < 1:
            raise ValueError("patch_sizes must have at least one element")
        if any(p < 1 for p in self.patch_sizes):
            raise ValueError("all patch sizes must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.e_layers <= 0:
            raise ValueError("e_layers must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class CARDConfig:
    d_model: int = 128
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 256
    patch_len: int = 16
    stride: int = 8
    dropout: float = 0.1
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
        if self.patch_len <= 0:
            raise ValueError("patch_len must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class RLinearConfig:
    individual: bool = True

    def validate(self) -> None:
        pass  # no numeric constraints


@dataclass
class ModernTCNConfig:
    patch_size: int = 8
    patch_stride: int = 4
    d_model: int = 128
    kernel_size: int = 51
    e_layers: int = 3
    d_ff_ratio: int = 4
    dropout: float = 0.05
    revin: bool = True

    def validate(self) -> None:
        if self.patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        if self.patch_stride < 1:
            raise ValueError("patch_stride must be >= 1")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        if self.e_layers < 1:
            raise ValueError("e_layers must be >= 1")
        if self.d_ff_ratio < 1:
            raise ValueError("d_ff_ratio must be >= 1")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class MambaForecasterConfig:
    d_model: int = 64
    d_state: int = 16
    e_layers: int = 2
    dropout: float = 0.05
    revin: bool = True

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.d_state <= 0:
            raise ValueError("d_state must be positive")
        if self.e_layers < 1:
            raise ValueError("e_layers must be >= 1")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class DishTSConfig:
    d_model: int = 256
    n_heads: int = 8
    e_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    dish_hidden: int = 64

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
        if self.dish_hidden <= 0:
            raise ValueError("dish_hidden must be positive")


@dataclass
class FiLMConfig:
    d_order: int = 32
    n_lowpass: int = 2
    d_ff: int = 256
    dropout: float = 0.05
    revin: bool = True

    def validate(self) -> None:
        if self.d_order < 1:
            raise ValueError("d_order must be >= 1")
        if self.n_lowpass < 1:
            raise ValueError("n_lowpass must be >= 1")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class TFTConfig:
    d_model: int = 128
    n_heads: int = 4
    num_lstm_layers: int = 2
    dropout: float = 0.1

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.num_lstm_layers < 1:
            raise ValueError("num_lstm_layers must be >= 1")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class MICNConfig:
    d_model: int = 64
    num_scales: int = 3
    kernel_size: int = 5
    dropout: float = 0.05
    revin: bool = True

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_scales < 1:
            raise ValueError("num_scales must be >= 1")
        if self.kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class NSTransformerConfig:
    d_model: int = 256
    n_heads: int = 8
    e_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1

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


@dataclass
class NormalizingFlowConfig:
    d_model: int = 256
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    revin: bool = True
    num_samples: int = 50
    flow_layers: int = 6
    flow_hidden: int = 128

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
        if self.flow_layers < 1:
            raise ValueError("flow_layers must be >= 1")
        if self.flow_hidden <= 0:
            raise ValueError("flow_hidden must be positive")


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
        ("NormalizingFlow", "Forecast"): NormalizingFlowConfig,
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
        ("ETSformer", "Forecast"): ETSformerConfig,
        ("ETSformer", "AnomalyDetection"): ETSformerConfig,
        ("ETSformer", "Imputation"): ETSformerConfig,
        ("ETSformer", "UEAClassification"): ETSformerConfig,
        ("NSTransformer", "Forecast"): NSTransformerConfig,
        ("NSTransformer", "AnomalyDetection"): NSTransformerConfig,
        ("NSTransformer", "Imputation"): NSTransformerConfig,
        ("NSTransformer", "UEAClassification"): NSTransformerConfig,
        ("MICN", "Forecast"): MICNConfig,
        ("MICN", "AnomalyDetection"): MICNConfig,
        ("MICN", "Imputation"): MICNConfig,
        ("MICN", "UEAClassification"): MICNConfig,
        ("TFT", "Forecast"): TFTConfig,
        ("TFT", "AnomalyDetection"): TFTConfig,
        ("TFT", "Imputation"): TFTConfig,
        ("TFT", "UEAClassification"): TFTConfig,
        ("FiLM", "Forecast"): FiLMConfig,
        ("FiLM", "AnomalyDetection"): FiLMConfig,
        ("FiLM", "Imputation"): FiLMConfig,
        ("FiLM", "UEAClassification"): FiLMConfig,
        ("DishTS", "Forecast"): DishTSConfig,
        ("DishTS", "AnomalyDetection"): DishTSConfig,
        ("DishTS", "Imputation"): DishTSConfig,
        ("DishTS", "UEAClassification"): DishTSConfig,
        ("MambaForecaster", "Forecast"): MambaForecasterConfig,
        ("MambaForecaster", "AnomalyDetection"): MambaForecasterConfig,
        ("MambaForecaster", "Imputation"): MambaForecasterConfig,
        ("MambaForecaster", "UEAClassification"): MambaForecasterConfig,
        ("ModernTCN", "Forecast"): ModernTCNConfig,
        ("ModernTCN", "AnomalyDetection"): ModernTCNConfig,
        ("ModernTCN", "Imputation"): ModernTCNConfig,
        ("ModernTCN", "UEAClassification"): ModernTCNConfig,
        ("RLinear", "Forecast"): RLinearConfig,
        ("RLinear", "AnomalyDetection"): RLinearConfig,
        ("RLinear", "Imputation"): RLinearConfig,
        ("RLinear", "UEAClassification"): RLinearConfig,
        ("FilterNet", "Forecast"): FilterNetConfig,
        ("FilterNet", "AnomalyDetection"): FilterNetConfig,
        ("FilterNet", "Imputation"): FilterNetConfig,
        ("FilterNet", "UEAClassification"): FilterNetConfig,
        ("CARD", "Forecast"): CARDConfig,
        ("CARD", "AnomalyDetection"): CARDConfig,
        ("CARD", "Imputation"): CARDConfig,
        ("CARD", "UEAClassification"): CARDConfig,
        ("Pathformer", "Forecast"): PathformerConfig,
        ("Pathformer", "AnomalyDetection"): PathformerConfig,
        ("Pathformer", "Imputation"): PathformerConfig,
        ("Pathformer", "UEAClassification"): PathformerConfig,
        ("SMamba", "Forecast"): SMambaConfig,
        ("SMamba", "AnomalyDetection"): SMambaConfig,
        ("SMamba", "Imputation"): SMambaConfig,
        ("SMamba", "UEAClassification"): SMambaConfig,
        ("iMamba", "Forecast"): iMambaConfig,
        ("iMamba", "AnomalyDetection"): iMambaConfig,
        ("iMamba", "Imputation"): iMambaConfig,
        ("iMamba", "UEAClassification"): iMambaConfig,
        ("Basisformer", "Forecast"): BasisformerConfig,
        ("Basisformer", "AnomalyDetection"): BasisformerConfig,
        ("Basisformer", "Imputation"): BasisformerConfig,
        ("Basisformer", "UEAClassification"): BasisformerConfig,
        ("GCNForecaster", "Forecast"): GCNForecasterConfig,
        ("GCNForecaster", "AnomalyDetection"): GCNForecasterConfig,
        ("GCNForecaster", "Imputation"): GCNForecasterConfig,
        ("GCNForecaster", "UEAClassification"): GCNForecasterConfig,
        ("MoEForecaster", "Forecast"): MoEForecasterConfig,
        ("MoEForecaster", "AnomalyDetection"): MoEForecasterConfig,
        ("MoEForecaster", "Imputation"): MoEForecasterConfig,
        ("MoEForecaster", "UEAClassification"): MoEForecasterConfig,
        ("RetForecaster", "Forecast"): RetForecasterConfig,
        ("RetForecaster", "AnomalyDetection"): RetForecasterConfig,
        ("RetForecaster", "Imputation"): RetForecasterConfig,
        ("RetForecaster", "UEAClassification"): RetForecasterConfig,
        ("HarmonicForecaster", "Forecast"): HarmonicForecasterConfig,
        ("HarmonicForecaster", "AnomalyDetection"): HarmonicForecasterConfig,
        ("HarmonicForecaster", "Imputation"): HarmonicForecasterConfig,
        ("HarmonicForecaster", "UEAClassification"): HarmonicForecasterConfig,
        ("HDMixer", "Forecast"): HDMixerConfig,
        ("HDMixer", "AnomalyDetection"): HDMixerConfig,
        ("HDMixer", "Imputation"): HDMixerConfig,
        ("HDMixer", "UEAClassification"): HDMixerConfig,
        ("KANForecaster", "Forecast"): KANForecasterConfig,
        ("KANForecaster", "AnomalyDetection"): KANForecasterConfig,
        ("KANForecaster", "Imputation"): KANForecasterConfig,
        ("KANForecaster", "UEAClassification"): KANForecasterConfig,
        ("TSReservoir", "Forecast"): TSReservoirConfig,
        ("TSReservoir", "AnomalyDetection"): TSReservoirConfig,
        ("TSReservoir", "Imputation"): TSReservoirConfig,
        ("TSReservoir", "UEAClassification"): TSReservoirConfig,
        ("WaveletForecaster", "Forecast"): WaveletForecasterConfig,
        ("WaveletForecaster", "AnomalyDetection"): WaveletForecasterConfig,
        ("WaveletForecaster", "Imputation"): WaveletForecasterConfig,
        ("WaveletForecaster", "UEAClassification"): WaveletForecasterConfig,
        ("LinearAttentionForecaster", "Forecast"): LinearAttentionForecasterConfig,
        ("LinearAttentionForecaster", "AnomalyDetection"): LinearAttentionForecasterConfig,
        ("LinearAttentionForecaster", "Imputation"): LinearAttentionForecasterConfig,
        ("LinearAttentionForecaster", "UEAClassification"): LinearAttentionForecasterConfig,
        ("DualDecompForecaster", "Forecast"): DualDecompForecasterConfig,
        ("DualDecompForecaster", "AnomalyDetection"): DualDecompForecasterConfig,
        ("DualDecompForecaster", "Imputation"): DualDecompForecasterConfig,
        ("DualDecompForecaster", "UEAClassification"): DualDecompForecasterConfig,
        ("HyperForecaster", "Forecast"): HyperForecasterConfig,
        ("HyperForecaster", "AnomalyDetection"): HyperForecasterConfig,
        ("HyperForecaster", "Imputation"): HyperForecasterConfig,
        ("HyperForecaster", "UEAClassification"): HyperForecasterConfig,
        ("GATForecaster", "Forecast"): GATForecasterConfig,
        ("GATForecaster", "AnomalyDetection"): GATForecasterConfig,
        ("GATForecaster", "Imputation"): GATForecasterConfig,
        ("GATForecaster", "UEAClassification"): GATForecasterConfig,
        ("BiLSTMForecaster", "Forecast"): BiLSTMForecasterConfig,
        ("BiLSTMForecaster", "AnomalyDetection"): BiLSTMForecasterConfig,
        ("BiLSTMForecaster", "Imputation"): BiLSTMForecasterConfig,
        ("BiLSTMForecaster", "UEAClassification"): BiLSTMForecasterConfig,
        ("RandomFourierForecaster", "Forecast"): RandomFourierForecasterConfig,
        ("RandomFourierForecaster", "AnomalyDetection"): RandomFourierForecasterConfig,
        ("RandomFourierForecaster", "Imputation"): RandomFourierForecasterConfig,
        ("RandomFourierForecaster", "UEAClassification"): RandomFourierForecasterConfig,
        ("SparseTransformerForecaster", "Forecast"): SparseTransformerForecasterConfig,
        ("SparseTransformerForecaster", "AnomalyDetection"): SparseTransformerForecasterConfig,
        ("SparseTransformerForecaster", "Imputation"): SparseTransformerForecasterConfig,
        ("SparseTransformerForecaster", "UEAClassification"): SparseTransformerForecasterConfig,
        ("FourierMixerForecaster", "Forecast"): FourierMixerForecasterConfig,
        ("FourierMixerForecaster", "AnomalyDetection"): FourierMixerForecasterConfig,
        ("FourierMixerForecaster", "Imputation"): FourierMixerForecasterConfig,
        ("FourierMixerForecaster", "UEAClassification"): FourierMixerForecasterConfig,
        ("TemporalConvAttentionForecaster", "Forecast"): TemporalConvAttentionForecasterConfig,
        ("TemporalConvAttentionForecaster", "AnomalyDetection"): TemporalConvAttentionForecasterConfig,
        ("TemporalConvAttentionForecaster", "Imputation"): TemporalConvAttentionForecasterConfig,
        ("TemporalConvAttentionForecaster", "UEAClassification"): TemporalConvAttentionForecasterConfig,
        ("AdaptiveSpectralForecaster", "Forecast"): AdaptiveSpectralForecasterConfig,
        ("AdaptiveSpectralForecaster", "AnomalyDetection"): AdaptiveSpectralForecasterConfig,
        ("AdaptiveSpectralForecaster", "Imputation"): AdaptiveSpectralForecasterConfig,
        ("AdaptiveSpectralForecaster", "UEAClassification"): AdaptiveSpectralForecasterConfig,
        ("LRUForecaster", "Forecast"): LRUForecasterConfig,
        ("LRUForecaster", "AnomalyDetection"): LRUForecasterConfig,
        ("LRUForecaster", "Imputation"): LRUForecasterConfig,
        ("LRUForecaster", "UEAClassification"): LRUForecasterConfig,
        ("S4Forecaster", "Forecast"): S4ForecasterConfig,
        ("S4Forecaster", "AnomalyDetection"): S4ForecasterConfig,
        ("S4Forecaster", "Imputation"): S4ForecasterConfig,
        ("S4Forecaster", "UEAClassification"): S4ForecasterConfig,
        ("HyenaForecaster", "Forecast"): HyenaForecasterConfig,
        ("HyenaForecaster", "AnomalyDetection"): HyenaForecasterConfig,
        ("HyenaForecaster", "Imputation"): HyenaForecasterConfig,
        ("HyenaForecaster", "UEAClassification"): HyenaForecasterConfig,
        ("PrototypicalForecaster", "Forecast"): PrototypicalForecasterConfig,
        ("PrototypicalForecaster", "AnomalyDetection"): PrototypicalForecasterConfig,
        ("PrototypicalForecaster", "Imputation"): PrototypicalForecasterConfig,
        ("PrototypicalForecaster", "UEAClassification"): PrototypicalForecasterConfig,
        ("MultiscaleConvForecaster", "Forecast"): MultiscaleConvForecasterConfig,
        ("MultiscaleConvForecaster", "AnomalyDetection"): MultiscaleConvForecasterConfig,
        ("MultiscaleConvForecaster", "Imputation"): MultiscaleConvForecasterConfig,
        ("MultiscaleConvForecaster", "UEAClassification"): MultiscaleConvForecasterConfig,
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
