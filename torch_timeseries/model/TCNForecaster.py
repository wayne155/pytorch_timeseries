import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.tcn import TemporalConvNet
from torch_timeseries.nn.revin import RevIN


class TCNForecaster(nn.Module):
    """TCN-based multi-task time series model.

    Applies a Temporal Convolutional Network over the time dimension, then
    projects the last hidden state to the prediction horizon.  All input
    channels are processed jointly by the TCN (implicit channel mixing).

    Architecture:
        1. Optional RevIN instance normalization.
        2. :class:`TemporalConvNet`: ``(B, enc_in, seq_len) → (B, d_model, seq_len)``
        3. Last hidden state: ``(B, d_model)``
        4. Linear projection → ``(B, pred_len × enc_in)`` → ``(B, pred_len, enc_in)``
        5. Optional RevIN denormalization.

    Args:
        seq_len (int): Input sequence length.
        pred_len (int): Prediction horizon.
        enc_in (int): Number of input features (channels).
        d_model (int): TCN hidden channel width. Defaults to 64.
        num_levels (int): Number of ``TemporalBlock`` layers.
            Receptive field = ``1 + (kernel_size-1) × (2^num_levels - 1)``.
            Defaults to 4.
        kernel_size (int): Kernel size shared across all TCN levels. Defaults to 3.
        dropout (float): Dropout probability inside each ``TemporalBlock``.
            Defaults to 0.1.
        revin (bool): If ``True``, applies RevIN instance normalization before
            the TCN and denormalizes the output. Defaults to ``True``.
        output_prob (int): If > 0, replaces the forecasting head with a
            classification head that outputs ``output_prob`` class logits.
            Defaults to 0.

    Shape:
        - Input: ``(B, seq_len, enc_in)``
        - Output (forecast/reconstruction): ``(B, pred_len, enc_in)``
        - Output (classification): ``(B, output_prob)``

    Tasks: Forecasting, Imputation, Anomaly Detection, Classification.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        num_levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        revin: bool = True,
        output_prob: int = 0,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.output_prob = output_prob
        self.revin = revin

        if revin and output_prob == 0:
            self.rev = RevIN(enc_in, affine=True)

        num_channels = [d_model] * num_levels
        self.tcn = TemporalConvNet(
            in_channels=enc_in,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        if output_prob > 0:
            self.head = nn.Linear(d_model, output_prob)
        else:
            self.head = nn.Linear(d_model, enc_in * pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        if self.revin and self.output_prob == 0:
            x = self.rev(x, "norm")

        x = x.transpose(1, 2)               # (B, C, L)
        out = self.tcn(x)                    # (B, d_model, L)
        last = out[:, :, -1]                 # (B, d_model)

        out = self.head(last)                # (B, output_prob) or (B, C * pred_len)

        if self.output_prob > 0:
            return out                       # (B, num_classes)

        out = out.view(out.size(0), self.pred_len, self.enc_in)  # (B, pred_len, C)

        if self.revin:
            out = self.rev(out, "denorm")

        return out
