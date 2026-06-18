"""Sequence patching utilities for Patch-Transformer models.

Reference: Nie et al., *A Time Series is Worth 64 Words: Long-Term Forecasting
with Transformers*, ICLR 2023 (PatchTST).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor


class Patcher(nn.Module):
    """Divide a time series into fixed-length overlapping patches.

    Operates in the **channels-last** format ``(B, L, C)`` that is the
    standard convention in this library.

    For sequence length ``L``, patch length ``p``, and stride ``s``, the
    number of patches is ``⌊(L + pad - p) / s⌋ + 1`` where ``pad`` is the
    end-padding applied when ``padding='end'``.

    Args:
        patch_len (int): Length of each patch (``P``).
        stride (int): Step size between patch start positions (``S``).
            Defaults to ``patch_len`` (non-overlapping).
        padding (str | int): How to pad the sequence so that the last
            patch is complete.

            - ``'end'`` (default): replicate the last time step to fill the
              final patch.
            - ``'none'``: no padding; last patch may be discarded if the
              sequence length is not evenly divisible.
            - an ``int``: pad with this constant value.

    Shape:
        - Input:  ``(B, L, C)``
        - Output: ``(B, num_patches, patch_len, C)``

    Attributes:
        num_patches (int | None): Number of output patches for the last
            input seen.  ``None`` until the first forward pass.

    Example::

        patcher = Patcher(patch_len=16, stride=8)
        x   = torch.randn(4, 96, 7)    # (B, L, C)
        out = patcher(x)               # (4, 12, 16, 7)  — 12 patches

        # Flatten patches for a linear model:
        flat = out.flatten(-2)         # (4, 12, 112)
    """

    def __init__(
        self,
        patch_len: int,
        stride: int | None = None,
        padding: str | int = "end",
    ) -> None:
        super().__init__()
        if stride is None:
            stride = patch_len
        assert patch_len >= 1
        assert stride >= 1
        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding
        self.num_patches: int | None = None

    def _pad(self, x: Tensor) -> Tensor:
        L = x.size(1)
        if self.padding == "none":
            return x
        # compute how many extra time steps are needed
        n_full = math.ceil(max(L - self.patch_len, 0) / self.stride)
        needed = n_full * self.stride + self.patch_len
        pad_len = max(0, needed - L)
        if pad_len == 0:
            return x
        if self.padding == "end":
            last = x[:, -1:, :]           # (B, 1, C)
            return torch.cat([x, last.expand(-1, pad_len, -1)], dim=1)
        # integer constant padding
        pad_tensor = torch.full(
            (x.size(0), pad_len, x.size(2)), float(self.padding),
            dtype=x.dtype, device=x.device,
        )
        return torch.cat([x, pad_tensor], dim=1)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, C)
        x = self._pad(x)
        # unfold along time dimension: (B, C, L_padded) → unfold → (B, C, N, P)
        x_t = x.transpose(1, 2)                           # (B, C, L_padded)
        patches = x_t.unfold(-1, self.patch_len, self.stride)  # (B, C, N, P)
        patches = patches.permute(0, 2, 3, 1)             # (B, N, P, C)
        self.num_patches = patches.size(1)
        return patches
