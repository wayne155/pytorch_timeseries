import torch
import torch.nn as nn

from torch_timeseries.nn.patching import Patcher
from torch_timeseries.nn.mlp import MixerBlock
from torch_timeseries.nn.revin import RevIN


class PatchMixer(nn.Module):
    """Patch-based MLP-Mixer forecaster.

    Segments the input into patches per channel, projects each patch to an
    embedding, then applies a stack of :class:`MixerBlock` layers to mix
    information across patch positions and embedding dimensions.  Each input
    channel is processed independently (channel-independent, like PatchTST).

    Architecture::

        Input (B, L, C)
          → RevIN norm
          → Patcher          → (B, N, patch_len, C)
          → permute/reshape  → (B×C, N, patch_len)
          → Linear           → (B×C, N, d_model)
          → MixerBlock × depth
          → flatten + Linear → (B×C, pred_len)
          → reshape          → (B, pred_len, C)
          → RevIN denorm

    Args:
        seq_len (int): Input sequence length.
        pred_len (int): Prediction horizon.
        enc_in (int): Number of input channels (features).
        patch_len (int): Length of each patch. Defaults to 16.
        stride (int): Patch stride. Defaults to 8.
        d_model (int): Patch embedding dimension. Defaults to 64.
        depth (int): Number of :class:`MixerBlock` layers. Defaults to 3.
        dropout (float): Dropout probability. Defaults to 0.1.
        revin (bool): Apply RevIN instance normalization. Defaults to ``True``.
        output_prob (int): If > 0, classify into ``output_prob`` classes.
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
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        depth: int = 3,
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

        self.patcher = Patcher(patch_len=patch_len, stride=stride, padding="end")
        # compute num_patches by running a dry forward
        with torch.no_grad():
            dummy = torch.zeros(1, seq_len, enc_in)
            _ = self.patcher(dummy)
            num_patches = self.patcher.num_patches

        self.num_patches = num_patches
        self.patch_proj = nn.Linear(patch_len, d_model)

        self.mixer_blocks = nn.Sequential(
            *[MixerBlock(num_patches, d_model, dropout=dropout) for _ in range(depth)]
        )

        flat_dim = num_patches * d_model
        if output_prob > 0:
            self.head = nn.Linear(flat_dim * enc_in, output_prob)
        else:
            self.head = nn.Linear(flat_dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        B, L, C = x.shape

        if self.revin and self.output_prob == 0:
            x = self.rev(x, "norm")

        patches = self.patcher(x)                          # (B, N, patch_len, C)
        N, P = patches.shape[1], patches.shape[2]

        # channel-independent: merge batch and channel
        patches = patches.permute(0, 3, 1, 2)             # (B, C, N, patch_len)
        patches = patches.reshape(B * C, N, P)             # (B*C, N, patch_len)

        z = self.patch_proj(patches)                       # (B*C, N, d_model)
        z = self.mixer_blocks(z)                           # (B*C, N, d_model)

        if self.output_prob > 0:
            z = z.reshape(B, C * N * z.shape[-1])         # (B, C*N*d_model)
            return self.head(z)                            # (B, output_prob)

        z = z.reshape(B * C, -1)                           # (B*C, N*d_model)
        out = self.head(z)                                 # (B*C, pred_len)
        out = out.reshape(B, C, self.pred_len)             # (B, C, pred_len)
        out = out.transpose(1, 2)                          # (B, pred_len, C)

        if self.revin:
            out = self.rev(out, "denorm")

        return out
