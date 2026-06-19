"""BiLSTMForecaster: Bidirectional LSTM with temporal additive attention.

Architecture comparison:
  RNNForecaster:     channel-joint (multi-channel input), last hidden state → head;
                     optional bidirectional, no attention.
  BiLSTMForecaster:  channel-independent (input_size=1), **Bahdanau additive
                     attention** over all T hidden states → context vector → head.

Key idea:
  Running a bidirectional LSTM over each variate independently gives a
  forward-backward hidden sequence h ∈ ℝ^{T × 2d}.  Additive attention computes
  a soft importance weight α_t ∈ (0,1) for every timestep:

      e_t  = v^T tanh(W_h h_t + b)          scalar energy
      α_t  = softmax({e_t})                  (T,)  attention over time
      ctx  = Σ_t α_t h_t                    (2d,) weighted context

  The context vector is then projected to the forecast horizon.  This lets the
  model focus on the most informative timesteps rather than relying solely on the
  last recurrent state.

Architecture::

    Input (B, T, C)
      → RevIN norm
      → CI reshape → (B·C, T, 1)
      → Bidirectional LSTM → (B·C, T, 2·d_model)
      → Additive attention → context (B·C, 2·d_model)
      → Linear head → (B·C, pred_len)
      → reshape → (B, pred_len, C)
      → RevIN denorm

Args:
    seq_len:    input lookback length T.
    pred_len:   forecast horizon.
    enc_in:     number of variates C.
    d_model:    LSTM hidden size (output is 2·d_model after bidirectional concat).
    num_layers: number of stacked LSTM layers.
    d_attn:     hidden size of the additive attention MLP.
    dropout:    dropout rate (applied between LSTM layers when num_layers > 1).
    revin:      use RevIN instance normalisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.revin import RevIN


class BiLSTMForecaster(nn.Module):
    """Bidirectional LSTM with Bahdanau temporal attention (channel-independent)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        num_layers: int = 2,
        d_attn: int = 32,
        dropout: float = 0.1,
        revin: bool = True,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in
        self._revin_flag = revin

        if revin:
            self.revin_layer = RevIN(enc_in, affine=True)

        rnn_drop = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_drop,
            bidirectional=True,
        )
        d_bi = d_model * 2  # bidirectional concatenation

        # Bahdanau additive attention: W_h projects h → d_attn; v scores it
        self.attn_w = nn.Linear(d_bi, d_attn, bias=True)
        self.attn_v = nn.Linear(d_attn, 1, bias=False)

        self.head = nn.Linear(d_bi, pred_len)
        self.drop = nn.Dropout(dropout)

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

        # Channel-independent: treat each variate as independent 1-D series
        x_ci = x.permute(0, 2, 1).reshape(B * C, T, 1)  # (BC, T, 1)

        # Bidirectional LSTM
        out, _ = self.lstm(x_ci)          # (BC, T, 2·d_model)
        out = self.drop(out)

        # Bahdanau additive attention over T timesteps
        energy = self.attn_v(torch.tanh(self.attn_w(out)))  # (BC, T, 1)
        alpha = F.softmax(energy, dim=1)                     # (BC, T, 1)
        context = (alpha * out).sum(dim=1)                   # (BC, 2·d_model)

        # Forecast head
        pred = self.head(context)                            # (BC, pred_len)
        pred = pred.reshape(B, C, self.pred_len).permute(0, 2, 1)  # (B, pred_len, C)

        if self._revin_flag:
            pred = self.revin_layer(pred, "denorm")
        return pred
