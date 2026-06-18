import torch
import torch.nn as nn


class _ResidualBlock(nn.Module):
    """MLP residual block: LayerNorm → Linear → Dropout → Linear + skip."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))
        return h + self.proj(x)


class TiDE(nn.Module):
    """TiDE — Time-series Dense Encoder (Das et al., Google TMLR 2023).

    A pure-MLP encoder-decoder for long-range multivariate forecasting.
    The look-back window is encoded via stacked residual MLP blocks;
    the latent representation is decoded independently for each future
    time step, then projected to the output dimension via a temporal
    decoder (optional per-step linear).

    Paper: *Long-term Forecasting with TiDE: Time-series Dense Encoder*
    https://arxiv.org/abs/2304.08424

    Args:
        seq_len (int): Look-back window length.
        pred_len (int): Prediction horizon.
        enc_in (int): Number of input channels.
        hidden_size (int): Hidden size of every residual block. Defaults to 256.
        num_encoder_layers (int): Number of encoder residual blocks. Defaults to 2.
        num_decoder_layers (int): Number of decoder residual blocks. Defaults to 2.
        decoder_output_dim (int): Latent dim per future step. Defaults to 8.
        dropout (float): Dropout rate. Defaults to 0.3.
        output_prob (int): If > 0, output class logits for classification.
            Defaults to 0.

    Tasks: Forecasting, Imputation, Anomaly Detection, Classification.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        hidden_size: int = 256,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        decoder_output_dim: int = 8,
        dropout: float = 0.3,
        output_prob: int = 0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.output_prob = output_prob

        # Feature projection: (B, L, C) → (B, L*C) → encode
        encoder_in = seq_len * enc_in
        self.encoder = nn.Sequential(*[
            _ResidualBlock(
                in_dim=encoder_in if i == 0 else hidden_size,
                hidden_dim=hidden_size,
                out_dim=hidden_size,
                dropout=dropout,
            )
            for i in range(num_encoder_layers)
        ])

        # Decoder: hidden → pred_len * decoder_output_dim
        decoder_out = pred_len * decoder_output_dim
        self.decoder = nn.Sequential(*[
            _ResidualBlock(
                in_dim=hidden_size if i == 0 else decoder_out,
                hidden_dim=hidden_size,
                out_dim=decoder_out,
                dropout=dropout,
            )
            for i in range(num_decoder_layers)
        ])

        # Temporal decoder: (B, pred_len, decoder_output_dim) → (B, pred_len, C)
        self.temporal_decoder = nn.Linear(decoder_output_dim, enc_in)

        # Residual connection from lookback → future (last seq step repeated)
        self.residual_proj = nn.Linear(enc_in, enc_in)

        if output_prob > 0:
            self.cls_head = nn.Linear(enc_in * pred_len, output_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        B, L, C = x.shape

        # Flatten and encode
        flat = x.reshape(B, L * C)              # (B, L*C)
        latent = self.encoder(flat)              # (B, hidden_size)

        # Decode to future steps
        dec = self.decoder(latent)               # (B, pred_len * D)
        dec = dec.reshape(B, self.pred_len, -1)  # (B, pred_len, D)
        out = self.temporal_decoder(dec)         # (B, pred_len, C)

        # Global residual: last observed value propagated forward
        out = out + self.residual_proj(x[:, -1:, :])  # broadcast over pred_len

        if self.output_prob > 0:
            out = self.cls_head(out.reshape(B, -1))
        return out
