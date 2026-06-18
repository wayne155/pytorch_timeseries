import torch
import torch.nn as nn

from torch_timeseries.nn.revin import RevIN


class RNNForecaster(nn.Module):
    """RNN-based multi-task time series model.

    A lightweight baseline that feeds the full input sequence through a
    stacked GRU or LSTM (or plain Elman RNN), then projects the final hidden
    state to the prediction horizon.

    Architecture::

        Input (B, L, C)
          → RevIN norm
          → RNN (B, L, C) → last hidden (B, hidden_size)
          → Linear → (B, pred_len × C)
          → reshape → (B, pred_len, C)
          → RevIN denorm

    Args:
        seq_len (int): Input sequence length.
        pred_len (int): Prediction horizon.
        enc_in (int): Number of input features (channels).
        hidden_size (int): RNN hidden state size. Defaults to 64.
        num_layers (int): Number of stacked RNN layers. Defaults to 2.
        rnn_type (str): One of ``'gru'``, ``'lstm'``, ``'rnn'``.
            Defaults to ``'gru'``.
        dropout (float): Dropout between layers (only when
            ``num_layers > 1``). Defaults to 0.1.
        bidirectional (bool): Use bidirectional RNN. Defaults to ``False``.
        revin (bool): Apply RevIN instance normalization. Defaults to ``True``.
        output_prob (int): If > 0, add a classification head.
            Defaults to 0.

    Shape:
        - Input: ``(B, seq_len, enc_in)``
        - Output (forecast): ``(B, pred_len, enc_in)``
        - Output (classification): ``(B, output_prob)``

    Tasks: Forecasting, Imputation, Anomaly Detection, Classification.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        rnn_type: str = "gru",
        dropout: float = 0.1,
        bidirectional: bool = False,
        revin: bool = True,
        output_prob: int = 0,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.output_prob = output_prob
        self.revin = revin
        self.bidirectional = bidirectional

        if revin and output_prob == 0:
            self.rev = RevIN(enc_in, affine=True)

        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}[rnn_type.lower()]
        self.rnn = rnn_cls(
            input_size=enc_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
        )

        d_out = hidden_size * (2 if bidirectional else 1)
        if output_prob > 0:
            self.head = nn.Linear(d_out, output_prob)
        else:
            self.head = nn.Linear(d_out, enc_in * pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        if self.revin and self.output_prob == 0:
            x = self.rev(x, "norm")

        out, hidden = self.rnn(x)    # out: (B, L, d_out)
        last = out[:, -1, :]         # (B, d_out)

        pred = self.head(last)       # (B, output_prob) or (B, C * pred_len)

        if self.output_prob > 0:
            return pred

        pred = pred.view(pred.size(0), self.pred_len, self.enc_in)  # (B, pred_len, C)

        if self.revin:
            pred = self.rev(pred, "denorm")

        return pred
