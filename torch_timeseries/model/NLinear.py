import torch
import torch.nn as nn


class NLinear(nn.Module):
    """NLinear — Normalized Linear forecaster (Zeng et al., AAAI 2023).

    Subtracts the last observed value from the input before applying a linear
    projection, then adds it back to the output.  This simple normalization
    makes the model robust to distribution shifts and is a strong baseline for
    long-term forecasting.

    Paper: *Are Transformers Effective for Time Series Forecasting?*
    https://ojs.aaai.org/index.php/AAAI/article/view/26317

    Args:
        seq_len (int): Input sequence length.
        pred_len (int): Prediction horizon length.
        enc_in (int): Number of input features (channels).
        individual (bool): If True, each channel has its own linear weights.
            Defaults to False (shared weights across channels).
        output_prob (int): If > 0, add a classification head outputting
            ``output_prob`` class logits. Defaults to 0 (regression only).

    Tasks: Forecasting, Imputation, Anomaly Detection, Classification.
    """

    def __init__(self, seq_len, pred_len, enc_in, individual: bool = False, output_prob=0):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual
        self.output_prob = output_prob

        if self.individual:
            self.Linear = nn.ModuleList(
                nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)
            )
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

        if self.output_prob > 0:
            self.projection = nn.Linear(enc_in * pred_len, self.output_prob)

    def forward(self, x):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = x.permute(0, 2, 1)

        if self.individual:
            output = torch.zeros(
                [x.size(0), x.size(1), self.pred_len],
                dtype=x.dtype,
                device=x.device,
            )
            for i in range(self.channels):
                output[:, i, :] = self.Linear[i](x[:, i, :])
        else:
            output = self.Linear(x)

        output = output.permute(0, 2, 1) + seq_last
        if self.output_prob > 0:
            output = output.reshape(output.shape[0], -1)
            output = self.projection(output)
        return output
