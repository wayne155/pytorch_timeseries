"""Temporal Convolutional Network (TCN) building blocks.

Reference: Bai et al., *An Empirical Evaluation of Generic Convolutional and
Recurrent Networks for Sequence Modeling*, 2018.
https://arxiv.org/abs/1803.01271
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CausalConv1d(nn.Module):
    """1-D causal convolution with dilation support.

    Pads the *left* side so the output at time ``t`` depends only on inputs
    ``≤ t``.  The output length equals the input length for any combination of
    ``kernel_size`` and ``dilation``.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size.
        dilation (int): Dilation factor. Defaults to 1.
        bias (bool): Whether to add a learnable bias. Defaults to True.

    Shape:
        - Input:  ``(B, C_in, L)``
        - Output: ``(B, C_out, L)``

    Example::

        conv = CausalConv1d(in_channels=32, out_channels=32,
                            kernel_size=3, dilation=2)
        x = torch.randn(4, 32, 96)
        out = conv(x)   # (4, 32, 96)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """One TCN residual block: two causal dilated convolutions + residual.

    Applies two :class:`CausalConv1d` layers (each followed by ReLU and
    dropout) with a skip connection.  When ``in_channels != out_channels`` a
    1×1 convolution is used for the residual projection.

    Args:
        in_channels (int): Input channel count.
        out_channels (int): Output channel count.
        kernel_size (int): Kernel size for both causal convolutions.
        dilation (int): Dilation factor.
        dropout (float): Dropout probability between the two conv layers.
            Defaults to 0.2.

    Shape:
        - Input:  ``(B, C_in, L)``
        - Output: ``(B, C_out, L)``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv1.conv.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.conv.weight, nonlinearity="relu")
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="relu")

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.conv1(x))
        out = self.drop(out)
        out = self.relu(self.conv2(out))
        out = self.drop(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network — a stack of :class:`TemporalBlock` layers.

    Exponentially increases dilation at each level (``2^0, 2^1, …, 2^{n-1}``)
    to give a receptive field that grows exponentially with depth.  With
    ``num_levels`` levels of kernel size ``k`` the receptive field covers
    ``1 + (k-1) × (2^0 + … + 2^{n-1}) = 1 + (k-1) × (2^n - 1)`` time steps.

    Args:
        in_channels (int): Number of input channels (features per time step).
        num_channels (list[int]): Output channels at each level.
            ``len(num_channels)`` determines the depth.
        kernel_size (int): Kernel size, shared across all levels. Defaults to 2.
        dropout (float): Dropout probability. Defaults to 0.2.

    Shape:
        - Input:  ``(B, C_in, L)``
        - Output: ``(B, num_channels[-1], L)``

    Example::

        tcn = TemporalConvNet(
            in_channels=7,
            num_channels=[64, 64, 64],
            kernel_size=3,
            dropout=0.1,
        )
        x   = torch.randn(4, 7, 96)       # (B, C, L) — channels-first!
        out = tcn(x)                       # (4, 64, 96)

    Note:
        The input follows PyTorch's Conv1d convention ``(B, C, L)`` — to apply
        on ``(B, L, C)`` time series tensors use ``.transpose(1, 2)`` before and
        after calling this module, or wrap it in a ``ChannelsLastTCN``.
    """

    def __init__(
        self,
        in_channels: int,
        num_channels: list[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            ch_in = in_channels if i == 0 else num_channels[i - 1]
            layers.append(
                TemporalBlock(ch_in, out_ch, kernel_size, dilation=2 ** i, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)
