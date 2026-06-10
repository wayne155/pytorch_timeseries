import torch
import torch.nn as nn


class NLinear(nn.Module):
    """Normalized Linear baseline.

    Subtract the last observed value before the linear projection and add it
    back afterward. This matches the LTSF-Linear NLinear baseline.
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
