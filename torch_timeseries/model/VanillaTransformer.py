import torch
import torch.nn as nn

from torch_timeseries.nn.attention import FullAttention, AttentionLayer
from torch_timeseries.nn.encoder import Encoder, EncoderLayer
from torch_timeseries.nn.embedding import DataEmbedding
from torch_timeseries.nn.revin import RevIN


class VanillaTransformer(nn.Module):
    """Encoder-only Transformer forecaster built from torch_timeseries building blocks.

    Embeds input with value + sinusoidal positional embeddings, runs N
    Transformer encoder layers (full self-attention + FFN + LayerNorm), then
    mean-pools over the sequence and projects to the output.

    For output_prob == 0 (forecasting): returns (B, pred_len, enc_in).
    For output_prob > 0 (classification/detection): returns (B, output_prob).

    Args:
        seq_len (int): Input sequence length.
        pred_len (int): Forecast horizon.
        enc_in (int): Number of input channels/features.
        d_model (int): Transformer hidden dimension. Defaults to 256.
        n_heads (int): Number of attention heads. Defaults to 4.
        e_layers (int): Number of encoder layers. Defaults to 3.
        d_ff (int): Feed-forward hidden dimension. Defaults to 512.
        dropout (float): Dropout probability. Defaults to 0.1.
        activation (str): Activation in FFN — ``'relu'`` or ``'gelu'``.
        revin (bool): Apply Reversible Instance Normalisation. Defaults to True.
        output_prob (int): If > 0, classification head size; disables RevIN.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 256,
        n_heads: int = 4,
        e_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        revin: bool = True,
        output_prob: int = 0,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.output_prob = output_prob
        self.revin = revin and output_prob == 0

        if self.revin:
            self.rev = RevIN(enc_in, affine=True)

        self.embedding = DataEmbedding(
            enc_in, d_model, embed_type="fixed", freq="h",
            dropout=dropout, time_embed=False,
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=dropout),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        if output_prob > 0:
            self.head = nn.Linear(d_model, output_prob)
        else:
            self.head = nn.Linear(d_model, enc_in * pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        if self.revin:
            x = self.rev(x, "norm")

        # embed: (B, L, d_model)
        out = self.embedding(x, None)
        out, _ = self.encoder(out)

        # mean-pool over sequence: (B, d_model)
        out = out.mean(dim=1)
        out = self.head(out)

        if self.output_prob > 0:
            return out

        # (B, pred_len, enc_in)
        out = out.view(out.size(0), self.pred_len, self.enc_in)
        if self.revin:
            out = self.rev(out, "denorm")
        return out
