"""N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting.

Reference: Oreshkin et al., "N-BEATS: Neural basis expansion analysis for
interpretable time series forecasting", ICLR 2020.
https://arxiv.org/abs/1905.10437

Supports two stack types:
  - "generic":      learned basis expansion — fully data-driven.
  - "trend":        polynomial basis of degree `degree_of_polynomial`.
  - "seasonality":  Fourier basis with `num_harmonics` components.

Usage:
    model = NBEATS(
        seq_len=96,
        pred_len=24,
        enc_in=7,
        stack_types=["trend", "seasonality", "generic"],
    )
    y_hat = model(x)   # (B, pred_len, enc_in)
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Basis functions
# ──────────────────────────────────────────────────────────────────────────────


def _trend_basis(pred_len: int, degree: int, device: torch.device) -> torch.Tensor:
    """Polynomial basis matrix ``(degree+1, pred_len)``."""
    t = torch.linspace(0, 1, pred_len, device=device)  # (T,)
    return torch.stack([t ** k for k in range(degree + 1)], dim=0)  # (D+1, T)


def _backcast_trend_basis(seq_len: int, degree: int, device: torch.device) -> torch.Tensor:
    t = torch.linspace(0, 1, seq_len, device=device)
    return torch.stack([t ** k for k in range(degree + 1)], dim=0)


def _seasonality_basis(pred_len: int, num_harmonics: int, device: torch.device) -> torch.Tensor:
    """Fourier basis matrix ``(2*H, pred_len)`` — cosine then sine terms."""
    t = torch.linspace(0, 1, pred_len, device=device)
    cos_terms = [torch.cos(2 * math.pi * h * t) for h in range(1, num_harmonics + 1)]
    sin_terms = [torch.sin(2 * math.pi * h * t) for h in range(1, num_harmonics + 1)]
    return torch.stack(cos_terms + sin_terms, dim=0)  # (2H, T)


def _backcast_seasonality_basis(seq_len: int, num_harmonics: int, device: torch.device):
    t = torch.linspace(0, 1, seq_len, device=device)
    cos_terms = [torch.cos(2 * math.pi * h * t) for h in range(1, num_harmonics + 1)]
    sin_terms = [torch.sin(2 * math.pi * h * t) for h in range(1, num_harmonics + 1)]
    return torch.stack(cos_terms + sin_terms, dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# Block
# ──────────────────────────────────────────────────────────────────────────────


class _NBEATSBlock(nn.Module):
    """Single N-BEATS block with shared FC trunk and basis expansion heads.

    Args:
        seq_len: input lookback window.
        pred_len: forecasting horizon.
        enc_in: number of channels (treated independently).
        hidden_size: width of the 4-layer FC trunk.
        stack_type: "generic", "trend", or "seasonality".
        degree_of_polynomial: only used when stack_type="trend".
        num_harmonics: only used when stack_type="seasonality".
        expansion_coefficient_dim: theta dimension for "generic".
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        hidden_size: int = 256,
        stack_type: str = "generic",
        degree_of_polynomial: int = 3,
        num_harmonics: int = 1,
        expansion_coefficient_dim: int = 32,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.stack_type = stack_type
        self.degree_of_polynomial = degree_of_polynomial
        self.num_harmonics = num_harmonics

        # Shared FC trunk (4 layers, channel-independent: input = seq_len)
        self.fc = nn.Sequential(
            nn.Linear(seq_len, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )

        if stack_type == "generic":
            theta_b_dim = expansion_coefficient_dim
            theta_f_dim = expansion_coefficient_dim
            self.basis_f = nn.Linear(theta_f_dim, pred_len, bias=False)
            self.basis_b = nn.Linear(theta_b_dim, seq_len, bias=False)
        elif stack_type == "trend":
            theta_b_dim = degree_of_polynomial + 1
            theta_f_dim = degree_of_polynomial + 1
            # Fixed polynomial basis — registered as buffer
            self.register_buffer("_trend_f", None)   # lazy init on forward
            self.register_buffer("_trend_b", None)
        elif stack_type == "seasonality":
            theta_b_dim = 2 * num_harmonics
            theta_f_dim = 2 * num_harmonics
            self.register_buffer("_season_f", None)
            self.register_buffer("_season_b", None)
        else:
            raise ValueError(f"stack_type must be 'generic', 'trend', or 'seasonality', got '{stack_type}'")

        self.theta_f_head = nn.Linear(hidden_size, theta_f_dim, bias=False)
        self.theta_b_head = nn.Linear(hidden_size, theta_b_dim, bias=False)

    def _get_basis(self):
        dev = next(self.parameters()).device
        if self.stack_type == "trend":
            if self._trend_f is None or self._trend_f.device != dev:
                self._trend_f = _trend_basis(self.pred_len, self.degree_of_polynomial, dev)
                self._trend_b = _backcast_trend_basis(self.seq_len, self.degree_of_polynomial, dev)
            return self._trend_f, self._trend_b
        elif self.stack_type == "seasonality":
            if self._season_f is None or self._season_f.device != dev:
                self._season_f = _seasonality_basis(self.pred_len, self.num_harmonics, dev)
                self._season_b = _backcast_seasonality_basis(self.seq_len, self.num_harmonics, dev)
            return self._season_f, self._season_b
        return None, None

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: ``(B, seq_len, enc_in)``

        Returns:
            backcast ``(B, seq_len, enc_in)``, forecast ``(B, pred_len, enc_in)``
        """
        B, T, N = x.shape
        # Process each channel independently: reshape to (B*N, T)
        x_flat = x.permute(0, 2, 1).reshape(B * N, T)   # (B*N, seq_len)

        h = self.fc(x_flat)                               # (B*N, hidden)
        theta_f = self.theta_f_head(h)                    # (B*N, theta_f)
        theta_b = self.theta_b_head(h)                    # (B*N, theta_b)

        if self.stack_type == "generic":
            forecast = self.basis_f(theta_f)              # (B*N, pred_len)
            backcast = self.basis_b(theta_b)              # (B*N, seq_len)
        else:
            basis_f, basis_b = self._get_basis()          # (theta, pred_len) / (theta, seq_len)
            forecast = theta_f @ basis_f                  # (B*N, pred_len)
            backcast = theta_b @ basis_b                  # (B*N, seq_len)

        forecast = forecast.reshape(B, N, self.pred_len).permute(0, 2, 1)  # (B, pred_len, N)
        backcast = backcast.reshape(B, N, T).permute(0, 2, 1)              # (B, seq_len, N)
        return backcast, forecast


# ──────────────────────────────────────────────────────────────────────────────
# Stack
# ──────────────────────────────────────────────────────────────────────────────


class _NBEATSStack(nn.Module):
    def __init__(self, num_blocks: int, **block_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([_NBEATSBlock(**block_kwargs) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor):
        stack_forecast = 0.0
        for block in self.blocks:
            backcast, forecast = block(x)
            x = x - backcast          # doubly residual connection
            stack_forecast = stack_forecast + forecast
        return x, stack_forecast


# ──────────────────────────────────────────────────────────────────────────────
# N-BEATS
# ──────────────────────────────────────────────────────────────────────────────


class NBEATS(nn.Module):
    """N-BEATS: Neural Basis Expansion Analysis for Time Series Forecasting.

    Args:
        seq_len: lookback window length.
        pred_len: forecast horizon.
        enc_in: number of variates (channels); each processed independently.
        stack_types: list of stack types — any combination of
            ``"generic"``, ``"trend"``, ``"seasonality"``.
        num_blocks: number of blocks per stack.
        hidden_size: FC trunk width per block.
        expansion_coefficient_dim: theta dimension for generic stacks.
        degree_of_polynomial: polynomial degree for trend stacks.
        num_harmonics: Fourier harmonics for seasonality stacks.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        stack_types: Optional[List[str]] = None,
        num_blocks: int = 3,
        hidden_size: int = 256,
        expansion_coefficient_dim: int = 32,
        degree_of_polynomial: int = 3,
        num_harmonics: int = 1,
    ):
        super().__init__()
        if stack_types is None:
            stack_types = ["generic", "generic", "generic"]

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.stack_types = stack_types

        self.stacks = nn.ModuleList([
            _NBEATSStack(
                num_blocks=num_blocks,
                seq_len=seq_len,
                pred_len=pred_len,
                enc_in=enc_in,
                hidden_size=hidden_size,
                stack_type=stype,
                expansion_coefficient_dim=expansion_coefficient_dim,
                degree_of_polynomial=degree_of_polynomial,
                num_harmonics=num_harmonics,
            )
            for stype in stack_types
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, seq_len, enc_in)``.

        Returns:
            Forecast ``(B, pred_len, enc_in)``.
        """
        total_forecast = 0.0
        residual = x
        for stack in self.stacks:
            residual, stack_forecast = stack(residual)
            total_forecast = total_forecast + stack_forecast
        return total_forecast
