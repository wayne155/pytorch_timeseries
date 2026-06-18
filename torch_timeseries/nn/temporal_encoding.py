"""Reusable temporal encoding / positional embedding modules.

All modules follow a common convention:
  - Input  ``tau``: ``(B, L)`` float tensor of absolute time positions
    (e.g. ``torch.arange(L).expand(B, L)`` for sequential index,
    or Unix timestamps normalised to [0, 1]).
  - Output: ``(B, L, d_model)`` float tensor ready to be added to or
    concatenated with value embeddings.

Exceptions:
  - :class:`RotaryEmbedding` operates on Q/K pairs inside an attention layer
    and does not follow the (B, L) → (B, L, d) convention.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor


class Time2Vec(nn.Module):
    """Time2Vec — learnable periodic + trend time encoding (Kazemi et al., 2019).

    Represents a scalar time value ``τ`` as a ``k``-dimensional vector:

    .. math::

        \\text{t2v}(\\tau)_i =
        \\begin{cases}
          \\omega_i \\tau + \\phi_i & i = 0 \\quad (\\text{linear trend}) \\\\
          \\sin(\\omega_i \\tau + \\phi_i) & 1 \\le i < k \\quad (\\text{periodic})
        \\end{cases}

    The frequencies ``ω`` and phases ``φ`` are learnable parameters, so the
    model discovers which periods are relevant for the task.

    Paper: *Time2Vec: Learning a Vector Representation of Time.*
    https://arxiv.org/abs/1907.05321

    Args:
        k (int): Output embedding dimension (1 trend + k-1 periodic components).

    Shape:
        - Input:  ``(B, L)`` — scalar time positions per batch and time step.
        - Output: ``(B, L, k)`` — Time2Vec encoding.

    Example::

        enc = Time2Vec(k=64)
        tau = torch.arange(96).float().unsqueeze(0).expand(4, -1)  # (4, 96)
        out = enc(tau)   # (4, 96, 64)
    """

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.W = nn.Parameter(torch.randn(k))
        self.b = nn.Parameter(torch.randn(k))

    def forward(self, tau: Tensor) -> Tensor:
        # tau: (B, L)
        proj = tau.unsqueeze(-1) * self.W + self.b   # (B, L, k)
        trend    = proj[..., :1]                       # (B, L, 1)  — linear
        periodic = torch.sin(proj[..., 1:])            # (B, L, k-1) — sinusoidal
        return torch.cat([trend, periodic], dim=-1)    # (B, L, k)


class LearnableFourierFeatures(nn.Module):
    """Learnable Fourier Features for temporal encoding (Li et al., NeurIPS 2021).

    Projects a scalar time position ``τ`` to a ``d_model``-dimensional vector
    using learned frequencies and phases:

    .. math::

        \\text{lff}(\\tau) = \\bigl[\\sin(\\omega_1 \\tau + \\phi_1), \\cos(\\omega_1 \\tau + \\phi_1),
                                    \\ldots\\bigr]

    Unlike fixed sinusoidal embeddings, the frequencies ``ω`` are trained
    end-to-end so the model can focus on relevant temporal scales.  The output
    dimension is always even (``d_model // 2`` sine + ``d_model // 2`` cosine).

    Paper: *Learnable Fourier Features for Multi-Dimensional Spatial Positional
    Encoding.*
    https://proceedings.neurips.cc/paper/2021/hash/8d86a35c80de5d09a5e5e8b9f54d7e15-Abstract.html

    Args:
        d_model (int): Output embedding dimension (must be even).

    Shape:
        - Input:  ``(B, L)`` — scalar time positions.
        - Output: ``(B, L, d_model)`` — Fourier encoding.

    Example::

        enc = LearnableFourierFeatures(d_model=128)
        tau = torch.linspace(0, 1, 96).unsqueeze(0).expand(4, -1)
        out = enc(tau)   # (4, 96, 128)
    """

    def __init__(self, d_model: int) -> None:
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        super().__init__()
        self.omega = nn.Parameter(torch.randn(d_model // 2))
        self.phi   = nn.Parameter(torch.zeros(d_model // 2))

    def forward(self, tau: Tensor) -> Tensor:
        # tau: (B, L)
        proj = tau.unsqueeze(-1) * self.omega + self.phi  # (B, L, d/2)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (B, L, d)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding — RoPE (Su et al., 2021).

    Encodes relative position information by rotating query and key vectors
    in attention by an angle proportional to their absolute position.
    Because the rotation is applied to Q and K before the dot-product, the
    attention score between position ``m`` and ``n`` depends only on
    ``m − n``, giving the model built-in relative-position awareness without
    any learned parameters.

    Unlike the other encoders in this module, :class:`RotaryEmbedding` is
    applied *inside* an attention layer, not to the value embeddings.

    Paper: *RoFormer: Enhanced Transformer with Rotary Position Embedding.*
    https://arxiv.org/abs/2104.09864

    Args:
        dim (int): Dimension of each attention head (``d_model // n_heads``).
        max_len (int): Maximum sequence length to pre-compute. Defaults to 5000.

    Shape:
        - ``q``, ``k``: ``(B, H, L, dim)`` — query / key tensors.
        - Returns rotated ``(q, k)`` with the same shape.

    Example::

        rope = RotaryEmbedding(dim=64)
        # inside a multi-head attention forward:
        q = q.view(B, L, H, D).transpose(1, 2)   # (B, H, L, D)
        k = k.view(B, L, H, D).transpose(1, 2)
        q, k = rope(q, k)
    """

    def __init__(self, dim: int, max_len: int = 5000) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int) -> None:
        if seq_len <= self._seq_len_cached:
            return
        self._seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)       # (L, dim/2)
        emb   = torch.cat([freqs, freqs], dim=-1)   # (L, dim)
        self._cos_cached = emb.cos()[None, None]    # (1, 1, L, dim)
        self._sin_cached = emb.sin()[None, None]

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        L = q.shape[-2]
        self._build_cache(L)
        cos = self._cos_cached[..., :L, :]
        sin = self._sin_cached[..., :L, :]
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


class SinusoidalEmbedding(nn.Module):
    """Fixed sinusoidal positional embedding (Vaswani et al., NeurIPS 2017).

    The classic Transformer positional encoding: even dimensions use ``sin``,
    odd dimensions use ``cos``, with wavelengths forming a geometric sequence
    from ``2π`` to ``2π × 10000``.  No learnable parameters.

    Paper: *Attention Is All You Need.*
    https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

    Args:
        d_model (int): Embedding dimension (must be even).
        max_len (int): Maximum sequence length. Defaults to 5000.
        scale (bool): If True, multiply value embeddings by ``√d_model``
            before adding the positional encoding (as in the original paper).
            Defaults to False.

    Shape:
        - Input:  ``(B, L)`` — integer or float time positions (values ignored;
          only the length ``L`` matters for the pre-computed table).
        - Output: ``(1, L, d_model)`` — broadcast-ready encoding.

    Example::

        pe = SinusoidalEmbedding(d_model=512)
        x  = torch.zeros(4, 96, 512)
        x  = x + pe(x)   # add positional encoding in-place
    """

    def __init__(self, d_model: int, max_len: int = 5000, scale: bool = False) -> None:
        super().__init__()
        self.scale = math.sqrt(d_model) if scale else 1.0

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = (torch.arange(0, d_model, 2, dtype=torch.float)
               * (-math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, d_model) or (B, L) — only L matters
        return self.pe[:, : x.size(1)] * self.scale
