# torch_timeseries/model/irregular/grud.py
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class GRUDCell(nn.Module):
    """Single GRU-D step with input and hidden-state exponential decay.

    Implements Eq. 1–3 from Che et al. (2018) "Recurrent Neural Networks for
    Multivariate Time Series with Missing Values."
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # GRU cell input: concatenation of [x_imputed, mask] → dim F*2
        self.gru_cell = nn.GRUCell(input_size * 2, hidden_size)
        # Learned decay rates (clamped ≥ 0 via relu during forward)
        self.log_gamma_x = nn.Parameter(torch.zeros(input_size))
        self.log_gamma_h = nn.Parameter(torch.zeros(hidden_size))
        # Learned global feature mean (substituted when no observation exists yet)
        self.x_mean = nn.Parameter(torch.zeros(input_size))

    def forward(
        self,
        x: Tensor,        # (B, F)  current raw input
        m: Tensor,        # (B, F)  1=observed, 0=missing
        delta: Tensor,    # (B, F)  time elapsed since last observation per feature
        x_last: Tensor,   # (B, F)  last observed value per feature
        h: Tensor,        # (B, H)  previous hidden state
    ) -> Tuple[Tensor, Tensor]:
        gamma_x = torch.exp(-torch.relu(self.log_gamma_x) * delta)           # (B, F)
        x_imputed = m * x + (1.0 - m) * (gamma_x * x_last + (1.0 - gamma_x) * self.x_mean)

        # Decay hidden state: use mean elapsed time across features
        delta_h = delta.mean(dim=-1, keepdim=True).expand_as(h)              # (B, H)
        gamma_h = torch.exp(-torch.relu(self.log_gamma_h) * delta_h)         # (B, H)
        h_decayed = gamma_h * h

        gru_in = torch.cat([x_imputed, m], dim=-1)                           # (B, F*2)
        h_new = self.gru_cell(gru_in, h_decayed)

        # Update last observed value only at observed positions
        x_last_new = m * x + (1.0 - m) * x_last
        return h_new, x_last_new


class GRUD(nn.Module):
    """GRU-D: Recurrent Neural Networks for Multivariate Time Series
    with Missing Values (Che et al., 2018).

    forward(x, t, mask) → (B, output_size) logits for classification.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cell = GRUDCell(input_size, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: Tensor,                    # (B, T, F)
        t: Tensor,                    # (B, T) normalized times
        mask: Tensor,                 # (B, T, F)
        x_time: Optional[Tensor] = None,  # ignored in Phase 1
    ) -> Tensor:                      # (B, output_size)
        B, T, F = x.shape
        h = x.new_zeros(B, self.cell.hidden_size)
        x_last = x.new_zeros(B, F)
        t_last = x.new_zeros(B, F)  # last observed time per feature

        for step in range(T):
            x_t = x[:, step, :]                              # (B, F)
            m_t = mask[:, step, :]                           # (B, F)
            t_t = t[:, step].unsqueeze(-1).expand(B, F)     # (B, F)

            delta = torch.clamp(t_t - t_last, min=0.0)      # (B, F)
            h, x_last = self.cell(x_t, m_t, delta, x_last, h)

            # Update last-observed time only at observed positions
            t_last = m_t * t_t + (1.0 - m_t) * t_last

        return self.fc(self.drop(h))
