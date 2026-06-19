"""MoEForecaster: Mixture-of-Experts forecaster for time-series.

Motivation:
  Real-world time series exhibit heterogeneous dynamics: some segments look
  like random walks, others like seasonal oscillations, and yet others follow
  sharp trend shifts.  A single monolithic model must compromise across all
  these regimes.

  Mixture of Experts (MoE) sidesteps this by maintaining K independent
  "expert" forecasters and a learned router that dynamically selects which
  experts to apply for each input.  The key properties are:
    • **Conditional computation**: each sample activates only k_active out of K
      experts (top-k sparse gating).
    • **Specialisation**: experts can specialise on sub-populations of inputs
      (e.g., trending vs. seasonal series).
    • **Soft weighting**: predictions are a weighted average of the top-k
      expert outputs, so the transition between experts is smooth.

  Architecture:
    1. **Router**: a lightweight MLP that reads per-series statistics
       (mean, std, trend slope, skewness) → K logits → top-k softmax weights.
    2. **Experts**: K independent linear forecasters (each: Linear(seq_len,
       pred_len), optionally with a 1-hidden-layer MLP) operating
       channel-independently.
    3. **Aggregation**: weighted sum of the top-k expert predictions.
    4. **RevIN** normalisation (applied channel-independently).

  Why linear experts?  They are lightweight (each is just one weight matrix),
  which keeps parameter count per expert low.  With K experts and top-k routing,
  the effective capacity is k * (seq_len * pred_len) per variate, controllable
  by tuning K and k.

Pipeline:
    x → RevIN → stats(x) → Router → top-k weights
              → [Expert_1(x), ..., Expert_K(x)] → weighted sum → denorm

Args:
    seq_len:      input lookback length.
    pred_len:     forecast horizon.
    enc_in:       number of variates.
    n_experts:    total number of expert networks K.
    k_active:     number of top-k experts to use per sample (k ≤ K).
    d_router:     hidden dimension of the router MLP.
    expert_type:  "linear" (single Linear) or "mlp" (two-layer MLP).
    d_ff:         hidden size of expert MLP (only used if expert_type="mlp").
    dropout:      dropout on router and expert outputs.
    revin:        use RevIN instance normalisation.
"""
from __future__ import annotations

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
# Expert
# ──────────────────────────────────────────────────────────────────────────────


class _LinearExpert(nn.Module):
    """Single linear expert: T → pred_len per variate."""

    def __init__(self, seq_len: int, pred_len: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*C, T) → (B*C, pred_len)"""
        return self.drop(self.linear(x))


class _MLPExpert(nn.Module):
    """Two-layer MLP expert: T → d_ff → pred_len per variate."""

    def __init__(self, seq_len: int, pred_len: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, pred_len),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*C, T) → (B*C, pred_len)"""
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────────────────────


class _Router(nn.Module):
    """Computes per-series routing statistics → K expert weights (top-k sparse)."""

    def __init__(self, n_experts: int, d_router: int, k_active: int, dropout: float):
        super().__init__()
        self.k_active = k_active
        # Input features: 4 time-series statistics (mean, std, trend, skew)
        n_stats = 4
        self.net = nn.Sequential(
            nn.Linear(n_stats, d_router),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_router, n_experts),
        )

    @staticmethod
    def _stats(x: torch.Tensor) -> torch.Tensor:
        """Compute 4 statistics per (B*C) series.

        Args:
            x: (B*C, T)
        Returns:
            (B*C, 4) — [mean, std, trend_slope, skew]
        """
        T = x.shape[1]
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True).clamp(1e-5)
        # Trend: coefficient of linear regression (normalised t)
        t = torch.linspace(-1, 1, T, device=x.device, dtype=x.dtype)
        x_cent = x - mean
        t_cent = t - t.mean()
        slope = (x_cent * t_cent).mean(-1, keepdim=True) / (t_cent.pow(2).mean() + 1e-8)
        # Skewness: E[(x-mu)^3] / std^3
        skew = ((x_cent / std) ** 3).mean(-1, keepdim=True)
        return torch.cat([mean, std, slope, skew], dim=-1)   # (B*C, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*C, T)
        Returns:
            weights: (B*C, K) — sparse top-k softmax weights
        """
        stats = self._stats(x)                         # (B*C, 4)
        logits = self.net(stats)                       # (B*C, K)

        if self.k_active >= logits.shape[-1]:
            return torch.softmax(logits, dim=-1)

        # Top-k sparse: zero out non-top-k logits, then softmax
        topk_vals, topk_idx = logits.topk(self.k_active, dim=-1)
        sparse = torch.full_like(logits, float("-inf"))
        sparse.scatter_(-1, topk_idx, topk_vals)
        return torch.softmax(sparse, dim=-1)            # (B*C, K)


# ──────────────────────────────────────────────────────────────────────────────
# MoEForecaster
# ──────────────────────────────────────────────────────────────────────────────


class MoEForecaster(nn.Module):
    """Mixture-of-Experts time-series forecaster with top-k sparse routing."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        n_experts: int = 8,
        k_active: int = 2,
        d_router: int = 32,
        expert_type: str = "linear",
        d_ff: int = 128,
        dropout: float = 0.1,
        revin: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.n_experts = n_experts
        self._revin_flag = revin

        if revin:
            self.revin_layer = _RevIN(enc_in)

        # Router
        k_active = min(k_active, n_experts)
        self.router = _Router(n_experts, d_router, k_active, dropout)

        # Expert bank
        if expert_type == "linear":
            self.experts = nn.ModuleList(
                [_LinearExpert(seq_len, pred_len, dropout) for _ in range(n_experts)]
            )
        else:
            self.experts = nn.ModuleList(
                [_MLPExpert(seq_len, pred_len, d_ff, dropout) for _ in range(n_experts)]
            )

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

        # Channel-independent
        x_ci = x.transpose(1, 2).reshape(B * C, T)       # (B*C, T)

        # Routing weights: (B*C, K)
        weights = self.router(x_ci)

        # Expert predictions: K × (B*C, pred_len)
        expert_preds = torch.stack(
            [exp(x_ci) for exp in self.experts], dim=1
        )                                                  # (B*C, K, pred_len)

        # Weighted combination: (B*C, K) @ (B*C, K, pred_len) → (B*C, pred_len)
        out = (weights.unsqueeze(-1) * expert_preds).sum(dim=1)  # (B*C, pred_len)

        out = out.reshape(B, C, self.pred_len).transpose(1, 2)   # (B, pred_len, C)

        if self._revin_flag:
            out = self.revin_layer(out, "denorm")
        return out
