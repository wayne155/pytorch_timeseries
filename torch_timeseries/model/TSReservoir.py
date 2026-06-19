"""TSReservoir: Echo State Network / Reservoir Computing for time-series forecasting.

Reference: Jaeger (2001) "The 'echo state' approach to analysing and training
recurrent neural networks", GMD Report 148.

Key idea:
  A Recurrent Neural Network (RNN) has two parts:
    (a) the dynamics — how the hidden state evolves over time
    (b) the readout — how predictions are extracted from the hidden state

  In reservoir computing, ONLY the readout is trained.  The recurrent weight
  matrix W_res (d_res × d_res) and input weight matrix W_in (d_res × 1) are
  initialised randomly and frozen — they act as a fixed *reservoir* that maps
  the input sequence into a rich, high-dimensional feature space via:

      h[t] = tanh( W_in · x[t]  +  W_res · h[t−1] )

  The *echo state property* (ESP) — which guarantees that the reservoir "forgets"
  its initial state and only represents the recent input history — holds when the
  spectral radius ρ(W_res) < 1.  We enforce this by scaling W_res after random
  initialisation:  W_res ← W_res × (target_ρ / actual_ρ).

  After running the reservoir for T steps, the readout sees either:
    • the final state h[T],  or
    • a linear combination of all states (mean-pool), controlled by `pool_states`.

  The readout is a single Linear layer; all gradients flow only through it.

  This architecture is fundamentally different from every trained-RNN forecaster
  because the recurrent dynamics are deterministic and fixed — the model cannot
  over-fit to spurious recurrent patterns.

Architecture:
  1. RevIN normalise.
  2. Channel-independent: (B, T, C) → (B*C, T).
  3. Run fixed reservoir for T steps.
  4. Pool states → (B*C, d_res).
  5. Learnable linear readout → (B*C, pred_len).
  6. RevIN denormalise.

Args:
    seq_len:          input lookback T.
    pred_len:         forecast horizon.
    enc_in:           number of variates.
    d_res:            reservoir size (width of fixed recurrent network).
    spectral_radius:  target spectral radius ρ of W_res (must be < 1 for ESP).
    input_scale:      scale for W_in weights.
    pool_states:      if True, mean-pool all T states; else use final state only.
    revin:            use RevIN normalisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


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
# TSReservoir
# ──────────────────────────────────────────────────────────────────────────────


class TSReservoir(nn.Module):
    """Echo State Network forecaster with fixed reservoir + linear readout."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_res: int = 256,
        spectral_radius: float = 0.9,
        input_scale: float = 0.1,
        pool_states: bool = True,
        revin: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_res = d_res
        self.pool_states = pool_states
        self._revin_flag = revin

        if revin:
            self.revin_layer = _RevIN(enc_in)

        # ── Fixed reservoir weights (not parameters — not trained) ────────────
        # W_in: (d_res, 1) — maps scalar input to reservoir
        W_in = torch.randn(d_res, 1) * input_scale
        self.register_buffer("W_in", W_in)

        # W_res: (d_res, d_res) — recurrent reservoir; scaled to desired ρ
        W_res = torch.randn(d_res, d_res)
        eigvals = torch.linalg.eigvals(W_res)
        actual_sr = eigvals.abs().max().item()
        W_res = W_res * (spectral_radius / max(actual_sr, 1e-8))
        self.register_buffer("W_res", W_res)

        # ── Learnable readout (only trained component) ────────────────────────
        self.readout = nn.Linear(d_res, pred_len)

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

        x_ci = x.transpose(1, 2).reshape(B * C, T)         # (BC, T)

        # ── Reservoir dynamics (no gradients through W_res, W_in) ─────────────
        h = torch.zeros(B * C, self.d_res, device=x.device, dtype=x.dtype)
        if self.pool_states:
            all_states: list[torch.Tensor] = []
        for t in range(T):
            u = x_ci[:, t : t + 1]                         # (BC, 1)
            h = torch.tanh(u @ self.W_in.T + h @ self.W_res.T)
            if self.pool_states:
                all_states.append(h)

        # ── Pool and read out ─────────────────────────────────────────────────
        if self.pool_states:
            feat = torch.stack(all_states, dim=1).mean(dim=1)  # (BC, d_res)
        else:
            feat = h                                            # (BC, d_res)

        out = self.readout(feat)                            # (BC, pred_len)
        out = out.reshape(B, C, self.pred_len).transpose(1, 2)  # (B, pred_len, C)

        if self._revin_flag:
            out = self.revin_layer(out, "denorm")
        return out
