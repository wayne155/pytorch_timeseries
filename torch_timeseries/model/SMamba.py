"""S-Mamba: Inter-variate Transformer + Intra-variate Mamba for time-series forecasting.

Reference: Wang et al., "S-Mamba: Inter-Series Dependency Modeling via Multi-
variable Correlation Crafter for Multivariate Time-Series Forecasting", ICLR 2025.

Key ideas:
  PatchTST showed that channel-independent (CI) patch transformers are strong
  but miss inter-variate dependencies.  iTransformer uses inverted-attention
  across variates but sacrifices temporal locality.  S-Mamba unifies both via
  a two-stage architecture:

  1. **Patch Tokenization (CI)**: Each variate is cut into non-overlapping
     patches.  These become patch tokens of dimension d_model.  Processing is
     channel-independent at this stage, just like PatchTST.

  2. **Intra-variate Mamba (per-channel SSM)**: For each variate, the sequence
     of n_patches tokens is processed by a Mamba (selective SSM) block.  This
     efficiently captures local→global temporal dynamics within a single channel
     thanks to the recurrent SSM formulation (O(T) complexity, long-range memory).

  3. **Inter-variate Transformer (cross-channel attention)**: After intra-variate
     encoding, we reorganise tokens so that the sequence dimension is the variate
     axis.  Standard multi-head self-attention is applied across variates, so
     each variate can attend to all others.  This captures cross-channel
     correlations (similar in spirit to iTransformer).

  4. **Output Projection**: The enriched patch representations are flattened and
     projected to pred_len.

  Pipeline:
    x → RevIN → Patch Embed  → Mamba(intra) → Transformer(inter)
              → flatten/pool → Linear → pred_len → denorm

  S-Mamba's Mamba follows the simplified "diagonal SSM" from Mamba-1:
    A_diag ∈ R^{d_state}  (log-parameterised, forced negative)
    Delta, B, C  are input-dependent projections.
    Discrete ZOH step; sequential scan (not parallel scan for clarity).

Args:
    seq_len:     input lookback length.
    pred_len:    forecast horizon.
    enc_in:      number of variates.
    d_model:     token dimension.
    d_state:     SSM state dimension.
    e_layers:    number of (Mamba + Transformer) layer pairs.
    n_heads:     attention heads for the inter-variate transformer.
    d_ff:        feed-forward hidden size.
    patch_len:   patch length.
    stride:      patch stride (default = patch_len for non-overlapping).
    dropout:     dropout rate.
    revin:       use RevIN instance normalisation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# RevIN
# ──────────────────────────────────────────────────────────────────────────────


class _RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._mean = x.mean(1, keepdim=True)
            self._std = x.std(1, keepdim=True).clamp(self.eps)
            x = (x - self._mean) / self._std
            return x * self.affine_weight + self.affine_bias
        else:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            return x * self._std + self._mean


# ──────────────────────────────────────────────────────────────────────────────
# Selective SSM (simplified Mamba-1 core)
# ──────────────────────────────────────────────────────────────────────────────


class _SelectiveSSM(nn.Module):
    """Input-dependent diagonal SSM with ZOH discretisation.

    For each position t, computes:
        Δ_t  = softplus(W_Δ x_t)          step size  (D,)
        B_t  = W_B x_t                    input proj  (N,)
        C_t  = W_C x_t                    output proj (N,)
        A_bar = exp(Δ_t * A_diag)         ZOH of diagonal A   (D, N)
        B_bar = Δ_t * B_t                 ZOH of B            (D, N)
        h_t  = A_bar * h_{t-1} + B_bar ⊙ x_t  (element-wise, broadcasting)
        y_t  = (C_t · h_t).sum(-1) + D_skip * x_t
    """

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Diagonal A: parameterised in log space, forced negative
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))

        # Input-dependent projections
        self.W_delta = nn.Linear(d_model, d_model, bias=True)
        self.W_B = nn.Linear(d_model, d_state, bias=False)
        self.W_C = nn.Linear(d_model, d_state, bias=False)

        # Skip connection (D in Mamba notation)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            y: (B, L, D)
        """
        B, L, D = x.shape
        N = self.d_state

        A = -torch.exp(self.A_log)                                # (D, N) negative
        delta = F.softplus(self.W_delta(x))                       # (B, L, D)
        Bx = self.W_B(x)                                          # (B, L, N)
        Cx = self.W_C(x)                                          # (B, L, N)

        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            dt = delta[:, t, :]                                    # (B, D)
            # ZOH: A_bar = exp(dt ⊙ A)  — shape (B, D, N)
            dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
            # ZOH: B_bar_t = dt_t ⊙ B_t — shape (B, D, N)
            dB = dt.unsqueeze(-1) * Bx[:, t, :].unsqueeze(1)
            h = dA * h + dB * x[:, t, :].unsqueeze(-1)
            y_t = (Cx[:, t, :].unsqueeze(1) * h).sum(-1)          # (B, D)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                                 # (B, L, D)
        y = y + self.D * x
        return y


class _MambaBlock(nn.Module):
    """Full Mamba block: expand → SSM+gate → contract, with residual."""

    def __init__(self, d_model: int, d_state: int, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, d_inner * 2)   # SSM branch + gate branch
        self.ssm = _SelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        z, gate = self.in_proj(x).chunk(2, dim=-1)
        z = self.ssm(z)
        h = z * F.silu(gate)
        h = self.drop(self.out_proj(h))
        return h + residual


# ──────────────────────────────────────────────────────────────────────────────
# Inter-variate Transformer layer
# ──────────────────────────────────────────────────────────────────────────────


class _InterVariateTransformer(nn.Module):
    """Self-attention across the variate axis (iTransformer-style)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, d_model)  — C variates as sequence tokens
        """
        B, C, D = x.shape
        # Attention
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.reshape(B, C, self.n_heads, self.d_head).transpose(1, 2)
        k = k.reshape(B, C, self.n_heads, self.d_head).transpose(1, 2)
        v = v.reshape(B, C, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = self.drop(torch.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, C, D)
        x = x + self.proj(out)
        # FF
        x = x + self.ff(self.norm2(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# S-Mamba
# ──────────────────────────────────────────────────────────────────────────────


class SMamba(nn.Module):
    """S-Mamba: Intra-variate Mamba + Inter-variate Transformer."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        d_state: int = 16,
        e_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 128,
        patch_len: int = 16,
        stride: int | None = None,
        dropout: float = 0.1,
        revin: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self._revin_flag = revin

        if revin:
            self.revin_layer = _RevIN(enc_in)

        if stride is None:
            stride = patch_len
        self.patch_len = patch_len
        self.stride = stride

        n_patches = math.ceil((seq_len - patch_len) / stride) + 1
        self.n_patches = n_patches
        pad_len = max(0, (n_patches - 1) * stride + patch_len - seq_len)
        self.pad_len = pad_len

        # Patch embedding
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.drop = nn.Dropout(dropout)

        # Paired layers: intra-variate Mamba + inter-variate Transformer
        self.intra_layers = nn.ModuleList(
            [_MambaBlock(d_model, d_state, dropout=dropout) for _ in range(e_layers)]
        )
        self.inter_layers = nn.ModuleList(
            [_InterVariateTransformer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )

        # Output projection: mean-pool patches → d_model → pred_len
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len)

    def _patchify(self, x_ci: torch.Tensor) -> torch.Tensor:
        """(B*C, T) → (B*C, n_patches, patch_len)"""
        if self.pad_len > 0:
            x_ci = F.pad(x_ci, (0, self.pad_len))
        return x_ci.unfold(-1, self.patch_len, self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)
        Returns:
            (B, pred_len, C)
        """
        B, T, C = x.shape

        if self._revin_flag:
            x = self.revin_layer(x, "norm")

        # ── Patch embed (channel-independent) ────────────────────────────────
        x_ci = x.transpose(1, 2).reshape(B * C, T)
        patches = self._patchify(x_ci)                             # (B*C, n_patches, patch_len)
        h = self.patch_embed(patches) + self.pos_embed             # (B*C, n_patches, d_model)
        h = self.drop(h)

        # ── Layer-wise: intra Mamba → inter Transformer ───────────────────────
        for intra, inter in zip(self.intra_layers, self.inter_layers):
            # Intra-variate: each channel processes its own patches independently
            h = intra(h)                                           # (B*C, n_patches, d_model)

            # Inter-variate: attend across channels per patch position
            # Reshape to (B, n_patches, C, d_model) → for each patch, cross-variate attn
            h_inter = h.reshape(B, C, self.n_patches, -1)         # (B, C, n_patches, d_model)
            # Apply cross-variate attention independently at each patch position
            # Flatten patch and batch together for efficiency:
            # (B, n_patches, C, d_model)
            h_inter = h_inter.permute(0, 2, 1, 3)                 # (B, n_patches, C, d_model)
            h_inter = h_inter.reshape(B * self.n_patches, C, -1)  # (B*n, C, d_model)
            h_inter = inter(h_inter)                               # (B*n, C, d_model)
            h_inter = h_inter.reshape(B, self.n_patches, C, -1)   # (B, n_patches, C, d_model)
            h_inter = h_inter.permute(0, 2, 1, 3).reshape(B * C, self.n_patches, -1)
            h = h_inter

        # ── Head: mean-pool patches → pred_len ───────────────────────────────
        h = self.head_norm(h)
        pooled = h.mean(dim=1)                                     # (B*C, d_model)
        out = self.head(pooled)                                    # (B*C, pred_len)
        out = out.reshape(B, C, self.pred_len).transpose(1, 2)    # (B, pred_len, C)

        if self._revin_flag:
            out = self.revin_layer(out, "denorm")
        return out
