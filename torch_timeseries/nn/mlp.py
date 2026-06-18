"""MLP and Mixer building blocks for time series models."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class FeedForward(nn.Module):
    """Two-layer MLP with a bottleneck expansion — the standard Transformer FFN.

    Applies: Linear → Activation → Dropout → Linear → Dropout.

    Args:
        d_model (int): Input and output dimension.
        d_ff (int): Hidden dimension (typically 2× or 4× ``d_model``).
            Defaults to ``4 * d_model``.
        activation (str | nn.Module): Activation function name (``'relu'``,
            ``'gelu'``, ``'silu'``) or an ``nn.Module`` instance.
            Defaults to ``'gelu'``.
        dropout (float): Dropout probability applied after both linear
            projections. Defaults to 0.1.

    Shape:
        - Input:  ``(*, d_model)``
        - Output: ``(*, d_model)``

    Example::

        ffn = FeedForward(d_model=256, d_ff=1024, activation='gelu')
        x   = torch.randn(4, 96, 256)
        out = ffn(x)   # (4, 96, 256)
    """

    _ACTIVATIONS = {
        "relu":  nn.ReLU,
        "gelu":  nn.GELU,
        "silu":  nn.SiLU,
        "tanh":  nn.Tanh,
    }

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        activation: str | nn.Module = "gelu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        if isinstance(activation, str):
            act_cls = self._ACTIVATIONS.get(activation.lower())
            if act_cls is None:
                raise ValueError(
                    f"Unknown activation '{activation}'. "
                    f"Choose from {list(self._ACTIVATIONS)}"
                )
            activation = act_cls()
        self.linear1  = nn.Linear(d_model, d_ff)
        self.act      = activation
        self.drop1    = nn.Dropout(dropout)
        self.linear2  = nn.Linear(d_ff, d_model)
        self.drop2    = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop2(self.linear2(self.drop1(self.act(self.linear1(x)))))


class MixerBlock(nn.Module):
    """Channel-mixing + time-mixing MLP block (Tolstikhin et al., NeurIPS 2021).

    Applies alternating 1-D MLPs to the *time* and *channel* axes with
    residual connections, following the MLP-Mixer architecture adapted for
    time series.

    Concretely, for input ``X ∈ ℝ^{B × T × C}``:

    1. **Time mixing** — transpose to ``(B, C, T)``, apply a shared linear
       layer over the ``T`` dimension, transpose back.
    2. **Channel mixing** — apply a shared linear layer over the ``C``
       dimension.

    Layer normalization and residual connections wrap each step.

    Args:
        seq_len (int): Input sequence length (``T``).
        d_model (int): Channel / feature dimension (``C``).
        d_ff_time (int | None): Hidden units for time-mixing MLP.
            Defaults to ``seq_len``.
        d_ff_channel (int | None): Hidden units for channel-mixing MLP.
            Defaults to ``d_model``.
        dropout (float): Dropout after each MLP. Defaults to 0.1.

    Shape:
        - Input:  ``(B, T, C)``
        - Output: ``(B, T, C)``

    Example::

        block = MixerBlock(seq_len=96, d_model=64)
        x   = torch.randn(4, 96, 64)
        out = block(x)   # (4, 96, 64)

    Reference:
        Tolstikhin et al., *MLP-Mixer: An all-MLP Architecture for Vision*,
        NeurIPS 2021.  Wang et al., *TSMixer: An All-MLP Architecture for
        Time Series Forecasting*, 2023.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        d_ff_time: int | None = None,
        d_ff_channel: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_ff_time is None:
            d_ff_time = seq_len
        if d_ff_channel is None:
            d_ff_channel = d_model

        self.norm1 = nn.LayerNorm(d_model)
        self.time_mix = nn.Sequential(
            nn.Linear(seq_len, d_ff_time),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff_time, seq_len),
            nn.Dropout(dropout),
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.chan_mix = nn.Sequential(
            nn.Linear(d_model, d_ff_channel),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff_channel, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Time mixing: operate on T dimension
        # x: (B, T, C)
        y = self.norm1(x)
        y = y.transpose(1, 2)          # (B, C, T)
        y = self.time_mix(y)           # (B, C, T)
        y = y.transpose(1, 2)          # (B, T, C)
        x = x + y

        # Channel mixing: operate on C dimension
        x = x + self.chan_mix(self.norm2(x))
        return x
